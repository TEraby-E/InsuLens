"""ROS 2 image detector backed by an Ultralytics YOLOv10 weight."""

import json
import os
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from .modeling import create_yolo
from .small_object import generate_tiles, weighted_box_fusion


DEFAULT_MODEL_NAME = "insulator_defect_yolov10s.pt"
AUTO_MODEL_PATHS = {"", "auto"}


def resolve_model_path(configured_path: str) -> Path:
    """Resolve an explicit model path or locate the workspace default model."""
    configured_path = configured_path.strip()
    if configured_path.lower() not in AUTO_MODEL_PATHS:
        return Path(configured_path).expanduser().resolve()

    candidates = []
    environment_path = os.environ.get("INSULENS_MODEL_PATH", "").strip()
    if environment_path:
        candidates.append(Path(environment_path).expanduser())

    search_starts = [Path.cwd(), Path(__file__).resolve().parent]
    for start in search_starts:
        for directory in (start, *start.parents):
            candidates.append(directory / "models" / DEFAULT_MODEL_NAME)

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    searched = ", ".join(str(path) for path in seen)
    raise FileNotFoundError(
        f"YOLOv10 weight {DEFAULT_MODEL_NAME} could not be located "
        "automatically. "
        f"Searched: {searched}. Set INSULENS_MODEL_PATH or pass "
        "--ros-args -p model_path:=/absolute/path/to/model.pt."
    )


class YoloV10Detector(Node):
    """Subscribe to camera frames and publish insulator detections."""

    def __init__(self) -> None:
        super().__init__("yolov10_insulator_detector")
        self.declare_parameter(
            "model_path",
            "auto",
        )
        self.declare_parameter("image_topic", "/insulens/camera/image_raw")
        self.declare_parameter("annotated_topic", "/insulens/detection_image")
        self.declare_parameter("detections_topic", "/insulens/detections")
        self.declare_parameter("confidence", 0.35)
        self.declare_parameter("flashover_inference_threshold", 0.30)
        self.declare_parameter("flashover_candidate_confidence", 0.05)
        self.declare_parameter("iou", 0.55)
        self.declare_parameter("image_size", 768)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("frame_stride", 1)
        self.declare_parameter(
            "defect_classes", ["broken", "pollution", "missing", "flashover"]
        )
        self.declare_parameter("save_defect_evidence", True)
        self.declare_parameter(
            "evidence_dir", "inspection_results"
        )
        self.declare_parameter("evidence_cooldown_sec", 2.0)
        self.declare_parameter("tiled_inference.enabled", False)
        self.declare_parameter("tiled_inference.tile_size", 1024)
        self.declare_parameter("tiled_inference.overlap", 0.2)
        self.declare_parameter("tiled_inference.fusion_iou", 0.55)

        configured_model_path = str(self.get_parameter("model_path").value)
        model_path = resolve_model_path(configured_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLOv10 weight not found: {model_path}. Train it with "
                "ros2 run insulens_perception train_yolov10 first, or pass "
                "--ros-args -p model_path:=/absolute/path/to/model.pt."
            )
        try:
            import torch
            # Imported for the CUDA availability check. create_yolo() below
            # registers the project-owned Coordinate Attention layer.
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch/Ultralytics is missing; run scripts/setup_env.sh"
            ) from exc

        requested_device = str(self.get_parameter("device").value)
        if (
            requested_device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            self.get_logger().warning(
                "CUDA is unavailable; falling back to CPU"
            )
            requested_device = "cpu"
        self.device = requested_device
        self.confidence = float(self.get_parameter("confidence").value)
        self.flashover_threshold = float(
            self.get_parameter("flashover_inference_threshold").value
        )
        self.flashover_candidate_confidence = float(
            self.get_parameter("flashover_candidate_confidence").value
        )
        self.iou = float(self.get_parameter("iou").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.frame_stride = max(
            1, int(self.get_parameter("frame_stride").value)
        )
        self.defect_classes = set(self.get_parameter("defect_classes").value)
        self.save_defect_evidence = bool(
            self.get_parameter("save_defect_evidence").value
        )
        self.evidence_dir = Path(
            self.get_parameter("evidence_dir").value
        ).expanduser()
        self.evidence_cooldown = float(
            self.get_parameter("evidence_cooldown_sec").value
        )
        self.last_evidence_time = 0.0
        self.frame_index = 0
        self.tiled_inference = bool(
            self.get_parameter("tiled_inference.enabled").value
        )
        self.tile_size = int(self.get_parameter("tiled_inference.tile_size").value)
        self.tile_overlap = float(
            self.get_parameter("tiled_inference.overlap").value
        )
        self.tile_fusion_iou = float(
            self.get_parameter("tiled_inference.fusion_iou").value
        )

        self.model = create_yolo(str(model_path))
        self.bridge = CvBridge()
        self.annotated_publisher = self.create_publisher(
            Image, self.get_parameter("annotated_topic").value, 2
        )
        self.detections_publisher = self.create_publisher(
            String, self.get_parameter("detections_topic").value, 10
        )
        self.defect_publisher = self.create_publisher(
            String, "/insulens/defect_alerts", 10
        )
        self.latency_publisher = self.create_publisher(
            Float32, "/insulens/inference_ms", 10
        )
        self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Loaded YOLOv10 weight {model_path} on {self.device}; "
            f"tiled inference={'enabled' if self.tiled_inference else 'disabled'}"
        )

    def _on_image(self, message: Image) -> None:
        self.frame_index += 1
        if self.frame_index % self.frame_stride:
            return
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        started = time.perf_counter()
        if self.tiled_inference:
            detections = self._predict_tiled(frame)
            annotated = self._draw_detections(frame, detections)
        else:
            detections, annotated = self._predict_frame(frame)
        latency_ms = (time.perf_counter() - started) * 1000.0

        defects = [item for item in detections if item["is_defect"]]
        payload = {
            "stamp": {
                "sec": message.header.stamp.sec,
                "nanosec": message.header.stamp.nanosec,
            },
            "frame_id": message.header.frame_id,
            "inference_ms": round(latency_ms, 3),
            "defect_detected": bool(defects),
            "detections": detections,
        }
        detection_message = String()
        detection_message.data = json.dumps(payload, ensure_ascii=False)
        self.detections_publisher.publish(detection_message)

        annotated_message = self.bridge.cv2_to_imgmsg(
            annotated, encoding="bgr8"
        )
        annotated_message.header = message.header
        self.annotated_publisher.publish(annotated_message)
        latency_message = Float32()
        latency_message.data = latency_ms
        self.latency_publisher.publish(latency_message)

        if defects:
            alert = dict(payload)
            alert["detections"] = defects
            alert_message = String()
            alert_message.data = json.dumps(alert, ensure_ascii=False)
            self.defect_publisher.publish(alert_message)
            self._save_evidence(annotated, alert)

    def _predict_frame(self, frame):
        """Run ordinary full-frame inference and retain Ultralytics annotation."""
        result = self.model.predict(
            source=frame,
            conf=min(self.confidence, self.flashover_candidate_confidence),
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        detections = self._detections_from_result(result)
        return detections, self._draw_detections(frame, detections)

    def _predict_tiled(self, frame):
        """Infer overlapping crops, restore their coordinates, then fuse duplicates."""
        height, width = frame.shape[:2]
        detections = []
        for tile in generate_tiles(width, height, self.tile_size, self.tile_overlap):
            crop = frame[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
            result = self.model.predict(
                source=crop,
                conf=min(self.confidence, self.flashover_candidate_confidence),
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]
            for detection in self._detections_from_result(result):
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                detection["bbox_xyxy"] = [
                    round(x1 + tile.x, 2),
                    round(y1 + tile.y, 2),
                    round(x2 + tile.x, 2),
                    round(y2 + tile.y, 2),
                ]
                detections.append(detection)
        return weighted_box_fusion(detections, self.tile_fusion_iou)

    def _detections_from_result(self, result):
        detections = []
        if result.boxes is None:
            return detections
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, scores, classes):
            class_name = str(result.names[int(class_id)])
            inference_reason = None
            if class_name in {"normal", "broken", "pollution", "missing"} and float(score) < self.flashover_threshold:
                class_name = "flashover"
                inference_reason = "outside-four-trained-class-confidence"
            detections.append(
                {
                    "class_id": -1 if class_name == "flashover" else int(class_id),
                    "class_name": class_name,
                    "is_defect": class_name in self.defect_classes,
                    "confidence": round(float(score), 6),
                    "inference_reason": inference_reason,
                    "bbox_xyxy": [round(float(value), 2) for value in box],
                }
            )
        return detections

    @staticmethod
    def _draw_detections(frame, detections):
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])
            color = (0, 0, 255) if detection["is_defect"] else (0, 200, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            cv2.putText(
                annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA,
            )
        return annotated

    def _save_evidence(self, annotated, alert) -> None:
        if not self.save_defect_evidence:
            return
        now = time.time()
        if now - self.last_evidence_time < self.evidence_cooldown:
            return
        self.last_evidence_time = now
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        suffix = f"{int((now % 1) * 1000):03d}"
        stem = f"defect_{stamp}_{suffix}"
        cv2.imwrite(str(self.evidence_dir / f"{stem}.jpg"), annotated)
        (self.evidence_dir / f"{stem}.json").write_text(
            json.dumps(alert, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = YoloV10Detector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
