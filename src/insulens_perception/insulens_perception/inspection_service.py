"""Framework-independent image/video inspection primitives for InsuLens Web.

The Web contract is model-driven: class names and category counts are derived
from the loaded checkpoint instead of a project-specific, fixed class schema.
A deterministic OpenCV fallback is retained for installation checks only. It
is never a substitute for a trained detector in field deployment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable
from uuid import uuid4

import cv2
import imageio_ffmpeg
import numpy as np


CLASS_PALETTE = (
    "#38D59A",
    "#EF6757",
    "#2F80ED",
    "#F2A93B",
    "#8A72E8",
    "#25B7C4",
    "#D665A5",
    "#7CA84B",
)


def normalize_model_classes(names: object) -> dict[int, str]:
    """Normalize Ultralytics-style ``names`` dictionaries or sequences."""
    if isinstance(names, Mapping):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            normalized[class_id] = str(value)
        return dict(sorted(normalized.items()))
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def category_color(category: str, index: int | None = None) -> str:
    """Return a stable CSS color for arbitrary model labels."""
    if index is None:
        digest = hashlib.sha256(category.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big")
    return CLASS_PALETTE[index % len(CLASS_PALETTE)]


def opencv_color(category: str, index: int | None = None) -> tuple[int, int, int]:
    color = category_color(category, index).lstrip("#")
    red, green, blue = (int(color[offset : offset + 2], 16) for offset in (0, 2, 4))
    return blue, green, red


def infer_report_category(
    category: str,
    confidence: float,
    threshold: float | None,
    candidate_classes: Iterable[str] = (),
    inferred_class: str | None = None,
) -> tuple[str, bool]:
    """Optionally map a weak model candidate to a configured report-only class."""
    candidates = set(candidate_classes)
    if (
        inferred_class
        and threshold is not None
        and category != inferred_class
        and (not candidates or category in candidates)
        and confidence < threshold
    ):
        return inferred_class, True
    return category, False


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    category: str
    confidence: float
    source: str = "full-frame"
    track_id: int | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


def iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    """Return intersection-over-union for xyxy boxes."""
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def deduplicate(detections: Iterable[Detection], threshold: float = 0.55) -> list[Detection]:
    """Class-aware NMS used to merge full-frame and tiled small-object results."""
    selected: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if not any(
            detection.category == kept.category and iou(detection.bbox, kept.bbox) >= threshold
            for kept in selected
        ):
            selected.append(detection)
    return selected


class InsulatorDetector:
    """YOLO inference with full-frame plus overlapping tiled inference."""

    def __init__(
        self,
        weights: str | Path | None = None,
        confidence: float = 0.35,
        image_size: int = 960,
        tile_size: int = 960,
        tile_overlap: float = 0.2,
        class_aliases: Mapping[str, str] | None = None,
        class_labels: Mapping[str, str] | None = None,
        inferred_class: str | None = None,
        inferred_class_threshold: float | None = None,
        inference_candidate_confidence: float | None = None,
    ) -> None:
        self.confidence = confidence
        self.image_size = image_size
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.class_aliases = {str(key): str(value) for key, value in (class_aliases or {}).items()}
        self.class_labels = {str(key): str(value) for key, value in (class_labels or {}).items()}
        self.inferred_class = inferred_class.strip() if inferred_class and inferred_class.strip() else None
        self.inferred_class_threshold = inferred_class_threshold
        self.inference_candidate_confidence = inference_candidate_confidence
        self.model = None
        self.backend = "opencv-fallback"
        self.model_task: str | None = None
        resolved = Path(weights).expanduser() if weights else None
        self.weights = resolved.resolve() if resolved and resolved.is_file() else resolved
        self.model_classes: dict[int, str] = {}
        if resolved and resolved.is_file():
            try:
                from .modeling import create_yolo

                self.model = create_yolo(str(resolved))
                self.backend = "yolo+tiled-small-object"
                self.model_task = str(getattr(self.model, "task", "detect"))
                self.model_classes = normalize_model_classes(getattr(self.model, "names", {}))
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                self.load_error = str(error)
            else:
                self.load_error = None
        else:
            self.load_error = "未提供可用的模型权重，当前仅初始化了 OpenCV 演示回退检测器。"

    def _map_category(self, category: str) -> str:
        return self.class_aliases.get(category, self.class_aliases.get(category.lower(), category))

    @property
    def class_names(self) -> tuple[str, ...]:
        """Mapped model classes in checkpoint order, without duplicates."""
        ordered: list[str] = []
        for raw_name in self.model_classes.values():
            name = self._map_category(raw_name)
            if name not in ordered:
                ordered.append(name)
        return tuple(ordered)

    def category_schema(self, extra_categories: Iterable[str] = ()) -> list[dict]:
        """Describe every category the API may return to a schema-free client."""
        schema: list[dict] = []
        seen: set[str] = set()
        for class_id, raw_name in self.model_classes.items():
            name = self._map_category(raw_name)
            if name in seen:
                continue
            seen.add(name)
            schema.append({
                "id": class_id,
                "name": name,
                "source_name": raw_name,
                "display_name": self.class_labels.get(name, self.class_labels.get(raw_name, name)),
                "color": category_color(name, len(schema)),
            })
        optional = [self.inferred_class] if self.inferred_class else []
        optional.extend(str(category) for category in extra_categories)
        for name in optional:
            if not name or name in seen:
                continue
            seen.add(name)
            schema.append({
                "id": None,
                "name": name,
                "source_name": None,
                "display_name": self.class_labels.get(name, name),
                "color": category_color(name, len(schema)),
            })
        return schema

    def empty_category_counts(self) -> dict[str, int]:
        return {item["name"]: 0 for item in self.category_schema()}

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def ready_for_inspection(self) -> bool:
        """Whether a loaded checkpoint exposes box-compatible class metadata."""
        return self.model_loaded and bool(self.model_classes) and self.model_task != "classify"

    @property
    def compatibility_error(self) -> str | None:
        if self.load_error:
            return self.load_error
        if not self.model_classes:
            return "模型没有暴露可用的类别元数据 names。"
        if self.model_task == "classify":
            return "当前巡检接口需要目标框，暂不支持仅输出整图类别的 classify 模型。"
        return None

    def model_status(self) -> dict:
        return {
            "backend": self.backend,
            "model_loaded": self.model_loaded,
            "ready_for_inspection": self.ready_for_inspection,
            "model_task": self.model_task,
            "model_path": str(self.weights) if self.weights else None,
            "model_classes": self.model_classes,
            "class_names": list(self.class_names),
            "class_schema": self.category_schema(),
            "class_mapping": {label: self._map_category(label) for label in self.model_classes.values()},
            "inference_policy": {
                "inferred_class": self.inferred_class,
                "threshold": self.inferred_class_threshold,
                "candidate_confidence": self.inference_candidate_confidence,
            },
            "load_error": self.load_error,
            "compatibility_error": self.compatibility_error,
        }

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []
        if self.model is None:
            return self._fallback_detect(image)
        detections = self._predict(image, 0, 0, "full-frame")
        height, width = image.shape[:2]
        if max(height, width) > self.tile_size:
            stride = max(1, int(self.tile_size * (1 - self.tile_overlap)))
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    crop = image[y : min(y + self.tile_size, height), x : min(x + self.tile_size, width)]
                    if crop.shape[0] >= 96 and crop.shape[1] >= 96:
                        detections.extend(self._predict(crop, x, y, "tile"))
        return deduplicate(detections)

    def _predict(self, image: np.ndarray, offset_x: int, offset_y: int, source: str) -> list[Detection]:
        prediction_confidence = self.confidence
        if self.inferred_class and self.inference_candidate_confidence is not None:
            prediction_confidence = min(prediction_confidence, self.inference_candidate_confidence)
        result = self.model.predict(
            image,
            conf=prediction_confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0]
        names = normalize_model_classes(getattr(result, "names", self.model_classes))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            raise ValueError(
                f"当前 Web 检测适配器需要边界框输出，模型任务 {self.model_task or 'unknown'} 未返回 boxes。"
            )
        output: list[Detection] = []
        for box in boxes:
            class_id = int(box.cls[0].item())
            raw_category = names.get(class_id, self.model_classes.get(class_id, f"class_{class_id}"))
            category = self._map_category(raw_category)
            confidence = round(float(box.conf[0].item()), 4)
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())

            category, category_inferred = infer_report_category(
                category,
                confidence,
                self.inferred_class_threshold,
                self.class_names,
                self.inferred_class,
            )
            detection_source = f"{source}+{self.inferred_class}-inferred" if category_inferred else source

            output.append(
                Detection(
                    (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                    category,
                    confidence,
                    detection_source,
                )
            )
        return output

    def _fallback_detect(self, image: np.ndarray) -> list[Detection]:
        """Find high-contrast elongated bodies for end-to-end demo verification."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output: list[Detection] = []
        area_limit = max(150, image.shape[0] * image.shape[1] * 0.0002)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width * height < area_limit:
                continue
            ratio = max(width, height) / max(1, min(width, height))
            if ratio < 1.25:
                continue
            output.append(Detection((x, y, x + width, y + height), "object", 0.5, "fallback"))
        return output


class IoUTracker:
    """Lightweight class-aware tracker for stable object IDs across video frames."""

    def __init__(self, min_iou: float = 0.3, max_missed: int = 15) -> None:
        self.min_iou, self.max_missed = min_iou, max_missed
        self._tracks: dict[int, dict] = {}
        self._all_tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: Iterable[Detection]) -> list[Detection]:
        remaining = list(detections)
        assigned: list[Detection] = []
        used: set[int] = set()
        for track_id, track in list(self._tracks.items()):
            choices = [
                (iou(track["bbox"], item.bbox), index, item)
                for index, item in enumerate(remaining)
                if index not in used and item.category == track["category"]
            ]
            best = max(choices, default=(0.0, -1, None), key=lambda item: item[0])
            if best[0] >= self.min_iou:
                _, index, item = best
                used.add(index)
                self._tracks[track_id].update(bbox=item.bbox, missed=0, confidence=item.confidence)
                self._all_tracks[track_id] = dict(self._tracks[track_id])
                assigned.append(Detection(item.bbox, item.category, item.confidence, item.source, track_id))
            else:
                track["missed"] += 1
        for index, item in enumerate(remaining):
            if index in used:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = {
                "bbox": item.bbox, "category": item.category, "confidence": item.confidence, "missed": 0
            }
            self._all_tracks[track_id] = dict(self._tracks[track_id])
            assigned.append(Detection(item.bbox, item.category, item.confidence, item.source, track_id))
        self._tracks = {key: value for key, value in self._tracks.items() if value["missed"] <= self.max_missed}
        return assigned

    @property
    def confirmed_tracks(self) -> dict[int, dict]:
        return dict(self._tracks)

    @property
    def all_tracks(self) -> dict[int, dict]:
        """Return every object track observed during the current inspection."""
        return dict(self._all_tracks)


def draw_detections(image: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    rendered = image.copy()
    for item in detections:
        x1, y1, x2, y2 = item.bbox
        color = opencv_color(item.category)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        label = f"#{item.track_id or '-'} {item.category} {item.confidence:.0%}"
        cv2.putText(rendered, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
    return rendered


class InspectionRunner:
    """Execute a complete image/video inspection job and generate auditable reports."""

    def __init__(self, detector: InsulatorDetector, output_root: str | Path) -> None:
        self.detector = detector
        self.output_root = Path(output_root)

    def inspect_image(self, source: str | Path) -> dict:
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError("无法读取图片文件。")
        started = time.perf_counter()
        detections = self.detector.detect(image)
        elapsed = time.perf_counter() - started
        job = self._job_directory()
        rendered = draw_detections(image, detections)
        output = job / "annotated.jpg"
        cv2.imwrite(str(output), rendered)
        summary = self._summary(
            job, "image", 1, detections, {index + 1: {"category": item.category, "confidence": item.confidence} for index, item in enumerate(detections)}, elapsed
        )
        summary["output_media"] = output.name
        return self._write_reports(job, summary)

    def inspect_video(
        self,
        source: str | Path,
        job_id: str | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError("无法读取视频文件或视频编解码器不可用。")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else 25.0
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("无法读取视频分辨率。")
        job = self._job_directory(job_id)
        output = job / "annotated.mp4"
        try:
            writer = imageio_ffmpeg.write_frames(
                str(output),
                (width, height),
                fps=fps,
                codec="libx264",
                pix_fmt_in="bgr24",
                pix_fmt_out="yuv420p",
                output_params=[
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-crf", "22",
                    "-preset", "veryfast",
                    "-movflags", "+faststart",
                ],
            )
            writer.send(None)
        except (OSError, RuntimeError) as error:
            capture.release()
            raise ValueError("无法创建浏览器兼容的 H.264 输出视频。") from error
        tracker, frames, elapsed = IoUTracker(), 0, 0.0
        observations: list[dict] = []
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                started = time.perf_counter()
                tracked = tracker.update(self.detector.detect(frame))
                elapsed += time.perf_counter() - started
                writer.send(np.ascontiguousarray(draw_detections(frame, tracked)).tobytes())
                observations.append({"frame": frames, "detections": [item.to_dict() for item in tracked]})
                frames += 1
                if on_progress:
                    counts = Counter(track["category"] for track in tracker.all_tracks.values())
                    class_schema = self.detector.category_schema(counts)
                    on_progress({
                        "status": "processing",
                        "frames_processed": frames,
                        "frames_total": expected_frames,
                        "current_frame": observations[-1],
                        "category_counts": {
                            item["name"]: counts.get(item["name"], 0) for item in class_schema
                        },
                        "class_schema": class_schema,
                    })
        finally:
            capture.release()
            writer.close()
        if frames == 0:
            output.unlink(missing_ok=True)
            raise ValueError("视频中没有可读取的画面帧。")
        summary = self._summary(job, "video", frames, [], tracker.all_tracks, elapsed)
        summary.update(output_media=output.name, frame_observations=observations)
        return self._write_reports(job, summary)

    def _job_directory(self, job_id: str | None = None) -> Path:
        name = job_id or f"inspection_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        if Path(name).name != name:
            raise ValueError("非法任务标识。")
        job = self.output_root / name
        job.mkdir(parents=True, exist_ok=False)
        return job

    def _summary(self, job: Path, media_type: str, frames: int, detections: list[Detection], tracks: dict, elapsed: float) -> dict:
        categories = Counter(track["category"] for track in tracks.values())
        class_schema = self.detector.category_schema(categories)
        counts = {item["name"]: categories.get(item["name"], 0) for item in class_schema}
        confidences = [track["confidence"] for track in tracks.values()] or [item.confidence for item in detections]
        return {
            "job_id": job.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "media_type": media_type,
            "backend": self.detector.backend,
            "frames": frames,
            "detection_total": sum(counts.values()),
            "category_counts": counts,
            "class_schema": class_schema,
            "average_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 4),
            "processing_seconds": round(elapsed, 4),
            "fps": round(frames / elapsed, 2) if elapsed else 0.0,
        }

    def _write_reports(self, job: Path, summary: dict) -> dict:
        json_path, csv_path = job / "inspection_report.json", job / "inspection_report.csv"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["巡检任务", summary["job_id"]])
            writer.writerow(["检测数量", summary["detection_total"]])
            writer.writerow(["平均置信度", summary["average_confidence"]])
            writer.writerow(["处理速度(FPS)", summary["fps"]])
            writer.writerow([])
            writer.writerow(["类别", "数量"])
            writer.writerows(summary["category_counts"].items())
        summary["report_json"] = json_path.name
        summary["report_csv"] = csv_path.name
        return summary
