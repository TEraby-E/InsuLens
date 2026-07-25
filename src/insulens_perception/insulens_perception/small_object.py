"""Dataset analysis and tiled inference primitives for small-object detection.

The functions in this module intentionally do not depend on ROS or Ultralytics so
their coordinate mapping and fusion behaviour can be tested on a developer host.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Tile:
    """A window in original-image coordinates."""

    x: int
    y: int
    width: int
    height: int


def iou_distance(boxes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return ``1 - IoU`` for width/height pairs anchored at the origin."""
    if boxes.ndim != 2 or boxes.shape[1] != 2:
        raise ValueError("boxes must have shape (N, 2)")
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError("centroids must have shape (K, 2)")
    inter = np.minimum(boxes[:, None, :], centroids[None, :, :]).prod(axis=2)
    union = (
        boxes.prod(axis=1)[:, None]
        + centroids.prod(axis=1)[None, :]
        - inter
    )
    return 1.0 - inter / np.maximum(union, 1e-12)


def kmeans_iou(
    boxes: np.ndarray, clusters: int, max_iter: int = 300, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster width/height samples with IoU distance and K-means++ seeding."""
    if not 1 <= clusters <= len(boxes):
        raise ValueError("clusters must be between 1 and the number of boxes")
    rng = np.random.default_rng(seed)
    centroids = [boxes[rng.integers(len(boxes))]]
    for _ in range(1, clusters):
        distances = iou_distance(boxes, np.asarray(centroids)).min(axis=1)
        probabilities = distances**2
        total = probabilities.sum()
        if total <= 1e-12:
            centroids.append(boxes[rng.integers(len(boxes))])
        else:
            centroids.append(boxes[rng.choice(len(boxes), p=probabilities / total)])
    centroids_array = np.asarray(centroids, dtype=np.float64)
    for _ in range(max_iter):
        assignment = iou_distance(boxes, centroids_array).argmin(axis=1)
        updated = np.asarray(
            [
                np.median(boxes[assignment == index], axis=0)
                if np.any(assignment == index)
                else centroid
                for index, centroid in enumerate(centroids_array)
            ]
        )
        if np.allclose(updated, centroids_array, atol=1e-6):
            break
        centroids_array = updated
    return centroids_array, assignment


def generate_tiles(
    image_width: int, image_height: int, tile_size: int, overlap: float
) -> list[Tile]:
    """Cover an image with overlapping windows without leaving an uncovered edge."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, round(tile_size * (1.0 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        positions = list(range(0, length - tile_size + 1, step))
        final = length - tile_size
        if positions[-1] != final:
            positions.append(final)
        return positions

    return [
        Tile(x, y, min(tile_size, image_width - x), min(tile_size, image_height - y))
        for y in starts(image_height)
        for x in starts(image_width)
    ]


def bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    """Calculate IoU for two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def weighted_box_fusion(
    detections: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """Fuse same-class overlapping boxes after tiled inference.

    The resulting confidence is the strongest confidence in a group; coordinates
    are confidence-weighted.  This keeps alert thresholds interpretable while
    avoiding duplicated border detections.
    """
    remaining = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    fused: list[dict[str, Any]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        retained = []
        for candidate in remaining:
            same_class = candidate["class_id"] == seed["class_id"]
            if same_class and bbox_iou(seed["bbox_xyxy"], candidate["bbox_xyxy"]) >= iou_threshold:
                group.append(candidate)
            else:
                retained.append(candidate)
        remaining = retained
        weights = np.asarray([item["confidence"] for item in group], dtype=np.float64)
        boxes = np.asarray([item["bbox_xyxy"] for item in group], dtype=np.float64)
        merged = dict(seed)
        merged["bbox_xyxy"] = np.average(boxes, axis=0, weights=weights).round(2).tolist()
        merged["confidence"] = round(float(weights.max()), 6)
        merged["tile_observations"] = len(group)
        fused.append(merged)
    return fused


def _resolve_split_images(data: dict[str, Any], data_path: Path) -> list[Path]:
    root = Path(data.get("path", data_path.parent)).expanduser()
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    sources = [data.get(name) for name in ("train", "val", "test") if data.get(name)]
    images: list[Path] = []
    for source in sources:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = root / source_path
        if source_path.is_file() and source_path.suffix == ".txt":
            images.extend(
                Path(line.strip()) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        elif source_path.is_dir():
            images.extend(
                path for path in source_path.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
    return sorted(set(images))


def analyse_yolo_dataset(
    data_yaml: Path, clusters: int = 6, input_size: int = 640
) -> dict[str, Any]:
    """Build anchor-free small-object recommendations from YOLO labels."""
    data_yaml = data_yaml.expanduser().resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    samples: list[tuple[int, float, float]] = []
    skipped_labels = 0
    for image_path in _resolve_split_images(data, data_yaml):
        label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if not label_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_labels += 1
            continue
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                skipped_labels += 1
                continue
            try:
                class_id = int(float(values[0]))
                box_width = float(values[3]) * width
                box_height = float(values[4]) * height
            except ValueError:
                skipped_labels += 1
                continue
            if box_width > 0.0 and box_height > 0.0:
                samples.append((class_id, box_width, box_height))
    if not samples:
        raise ValueError("No valid YOLO boxes were found for the configured dataset splits")
    dimensions = np.asarray([[width, height] for _, width, height in samples], dtype=np.float64)
    centroids, assignment = kmeans_iou(dimensions, min(clusters, len(dimensions)))
    short_sides = np.min(dimensions, axis=1)
    threshold = input_size / 64.0
    class_counts = Counter(class_id for class_id, _, _ in samples)
    class_stats = {}
    for class_id, count in sorted(class_counts.items()):
        class_boxes = dimensions[[index for index, item in enumerate(samples) if item[0] == class_id]]
        class_stats[str(class_id)] = {
            "name": str(names.get(class_id, class_id)),
            "boxes": count,
            "median_width_px": round(float(np.median(class_boxes[:, 0])), 2),
            "median_height_px": round(float(np.median(class_boxes[:, 1])), 2),
        }
    cluster_counts = Counter(assignment.tolist())
    p2_recommended = float(np.median(short_sides)) < threshold or float(np.mean(short_sides < threshold)) >= 0.2
    return {
        "dataset": str(data_yaml),
        "input_size": input_size,
        "box_count": len(samples),
        "skipped_labels": skipped_labels,
        "class_statistics": class_stats,
        "small_object": {
            "threshold_px": round(threshold, 2),
            "median_short_side_px": round(float(np.median(short_sides)), 2),
            "below_threshold_fraction": round(float(np.mean(short_sides < threshold)), 4),
            "p2_recommended": p2_recommended,
        },
        "clusters": [
            {
                "id": int(index),
                "median_width_px": round(float(centroid[0]), 2),
                "median_height_px": round(float(centroid[1]), 2),
                "box_count": int(cluster_counts[index]),
            }
            for index, centroid in enumerate(centroids)
        ],
        "recommendations": {
            "architecture": "P2 + Coordinate Attention" if p2_recommended else "Baseline P3/P4/P5 or P2 ablation",
            "tal_topk_candidates": [6, 10, 13],
            "note": "YOLOv10 is anchor-free; cluster results guide P2/TAL/augmentation choices and do not create anchors.",
        },
    }


def write_analysis_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable JSON and a concise Markdown appendix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "small_object_scale_report.json"
    markdown_path = output_dir / "small_object_scale_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    small = report["small_object"]
    rows = "\n".join(
        f"| {item['id']} | {item['median_width_px']} | {item['median_height_px']} | {item['box_count']} |"
        for item in report["clusters"]
    )
    markdown_path.write_text(
        "# 小目标尺度分析报告\n\n"
        f"- 有效标注框：{report['box_count']}\n"
        f"- 中位短边：{small['median_short_side_px']} px\n"
        f"- P2 建议阈值：{small['threshold_px']} px\n"
        f"- 小于阈值的标注比例：{small['below_threshold_fraction']:.2%}\n"
        f"- P2 检测头建议：{'建议启用' if small['p2_recommended'] else '作为消融项验证'}\n"
        "- TAL topk 候选：6 / 10 / 13\n\n"
        "| 簇 | 中位宽(px) | 中位高(px) | 标注框数 |\n|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        "YOLOv10 为 anchor-free 检测器；该报告不生成或替换 anchors。\n",
        encoding="utf-8",
    )
    return json_path, markdown_path
