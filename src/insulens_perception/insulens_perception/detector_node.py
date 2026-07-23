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
        self.declare_parameter("iou", 0.55)
        self.declare_parameter("image_size", 768)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("frame_stride", 1)
        self.declare_parameter("defect_classes", ["missing_disc"])
        self.declare_parameter("save_defect_evidence", True)
        self.declare_parameter(
            "evidence_dir", "inspection_results"
        )
        self.declare_parameter("evidence_cooldown_sec", 2.0)

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
            from ultralytics import YOLO
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

        self.model = YOLO(str(model_path))
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
            f"Loaded YOLOv10 weight {model_path} on {self.device}"
        )

    def _on_image(self, message: Image) -> None:
        self.frame_index += 1
        if self.frame_index % self.frame_stride:
            return
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        started = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        result = results[0]

        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(boxes, scores, classes):
                class_name = str(result.names[int(class_id)])
                detections.append(
                    {
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "is_defect": class_name in self.defect_classes,
                        "confidence": round(float(score), 6),
                        "bbox_xyxy": [round(float(value), 2) for value in box],
                    }
                )

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

        annotated = result.plot()
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
