"""Build and validate the four trained-class insulator YOLO dataset.

The five report classes are normal, broken, pollution, missing, and flashover.
Only the first four are trained; flashover remains a report-layer inference.
Every imported defect image must have explicit provenance. Pollution synthesis
uses a user-supplied real contamination texture and mask instead of drawn noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping

import cv2
import numpy as np
import yaml


TRAIN_CLASS_NAMES = ("normal", "broken", "pollution", "missing")
REPORT_CLASS_NAMES = (*TRAIN_CLASS_NAMES, "flashover")
CLASS_IDS = {name: index for index, name in enumerate(TRAIN_CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CPLID_SOURCE_URL = "https://github.com/InsulatorData/InsulatorDataSet"
CPLID_COMMIT = "1f6349f619237344d49905090ecf2704505394a4"
TLID_SOURCE_URL = "https://github.com/Caughyhzd/TLID-Dataset"
DRAEM_SOURCE_URL = "https://github.com/VitjanZ/DRAEM"
DTD_SOURCE_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/"
DTD_LICENSE = "research-use-only (per DTD dataset page)"
DTD_POLLUTION_TEXTURE_CLASSES = (
    "blotchy",
    "flecked",
    "freckled",
    "smeared",
    "sprinkled",
    "stained",
)


def parse_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO boxes and reject malformed or out-of-range coordinates."""
    if not path.is_file():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} must have five YOLO fields")
        class_id, *coordinates = fields
        values = tuple(float(value) for value in coordinates)
        if (
            int(class_id) != float(class_id)
            or not all(0 < value <= 1 for value in values[2:])
            or not all(0 <= value <= 1 for value in values[:2])
        ):
            raise ValueError(f"{path}:{line_number} has invalid YOLO coordinates")
        boxes.append((int(class_id), *values))
    return boxes


def write_yolo_labels(path: Path, boxes: Iterable[tuple[int, float, float, float, float]]) -> None:
    path.write_text(
        "".join(
            f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n"
            for class_id, cx, cy, width, height in boxes
        ),
        encoding="utf-8",
    )


def _source_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    if not image_dir.is_dir() or not label_dir.is_dir():
        return []
    return [
        (image_path, label_dir / f"{image_path.stem}.txt")
        for image_path in sorted(image_dir.iterdir())
        if image_path.suffix.lower() in IMAGE_SUFFIXES
        and (label_dir / f"{image_path.stem}.txt").is_file()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset_output(output_root: Path) -> None:
    for relative in ("images", "labels"):
        target = output_root / relative
        if target.exists():
            shutil.rmtree(target)
    for filename in ("data.yaml", "sources.jsonl", "dataset_metadata.json"):
        target = output_root / filename
        if target.exists():
            target.unlink()


def _copy_dataset(
    source_root: Path,
    output_root: Path,
    dataset_name: str,
    source_url: str,
    class_mapping: Mapping[int, int],
    manifest: list[dict],
    per_split: int | None = None,
) -> None:
    """Copy a YOLO source dataset and explicitly remap documented classes."""
    for split in ("train", "val"):
        pairs = _source_pairs(source_root / "images" / split, source_root / "labels" / split)
        if not pairs:
            raise ValueError(f"No image/label pairs in {source_root}/{split}")
        selected = pairs if per_split is None else pairs[: min(per_split, len(pairs))]
        image_dir, label_dir = output_root / "images" / split, output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in selected:
            source_boxes = parse_yolo_labels(label_path)
            unsupported = sorted({box[0] for box in source_boxes} - class_mapping.keys())
            if unsupported:
                raise ValueError(f"{label_path} uses unmapped {dataset_name} classes {unsupported}")
            labels = [(class_mapping[box[0]], *box[1:]) for box in source_boxes]
            if not labels:
                continue
            if cv2.imread(str(image_path)) is None:
                raise ValueError(f"Unable to decode source image {image_path}")
            stem = f"{dataset_name.lower()}_{image_path.stem}"
            target_image = image_dir / f"{stem}{image_path.suffix.lower()}"
            target_label = label_dir / f"{stem}.txt"
            if target_image.exists() or target_label.exists():
                raise ValueError(f"Duplicate target sample {stem}")
            shutil.copy2(image_path, target_image)
            write_yolo_labels(target_label, labels)
            manifest.append(
                {
                    "image": str(target_image.relative_to(output_root)),
                    "source_image": str(image_path.resolve()),
                    "source_label": str(label_path.resolve()),
                    "source_dataset": dataset_name,
                    "source_url": source_url,
                    "image_sha256": _sha256(image_path),
                    "origin": "source-copy",
                    "pixel_transform": "none",
                    "categories": sorted({TRAIN_CLASS_NAMES[label[0]] for label in labels}),
                }
            )


def synthesize_pollution_sample(
    image_path: Path,
    object_mask_path: Path,
    texture_path: Path,
    texture_mask_path: Path,
    output_path: Path,
    alpha: float = 0.55,
) -> tuple[float, float, float, float]:
    """Blend a real pollution texture through supplied masks and return its box.

    This follows DRAEM's texture-plus-mask anomaly synthesis principle while
    requiring real, traceable pollution texture input. No procedural dirt is
    generated. Validation images should remain real source copies.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    object_mask = cv2.imread(str(object_mask_path), cv2.IMREAD_GRAYSCALE)
    texture = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
    texture_mask = cv2.imread(str(texture_mask_path), cv2.IMREAD_GRAYSCALE)
    if any(item is None for item in (image, object_mask, texture, texture_mask)):
        raise ValueError("Unable to decode pollution synthesis inputs")
    height, width = image.shape[:2]
    object_mask = cv2.resize(object_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    texture = cv2.resize(texture, (width, height), interpolation=cv2.INTER_LINEAR)
    texture_mask = cv2.resize(texture_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    blend_mask = ((object_mask > 0) & (texture_mask > 0)).astype(np.uint8)
    if cv2.countNonZero(blend_mask) < 16:
        raise ValueError("Pollution and object masks do not overlap")
    feather = cv2.GaussianBlur(blend_mask.astype(np.float32), (0, 0), 2.0)
    feather = np.clip(feather * alpha, 0.0, 1.0)[..., None]
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_texture = cv2.cvtColor(texture, cv2.COLOR_BGR2LAB).astype(np.float32)
    blended = lab_image * (1.0 - feather) + lab_texture * feather
    output = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise ValueError(f"Unable to write {output_path}")
    x, y, box_width, box_height = cv2.boundingRect(blend_mask)
    return (
        (x + box_width / 2) / width,
        (y + box_height / 2) / height,
        box_width / width,
        box_height / height,
    )


def _grabcut_object_mask(
    image: np.ndarray,
    boxes: Iterable[tuple[int, float, float, float, float]],
) -> np.ndarray:
    """Estimate an insulator foreground mask from traceable YOLO boxes."""
    height, width = image.shape[:2]
    combined = np.zeros((height, width), dtype=np.uint8)
    for _, cx, cy, box_width, box_height in boxes:
        x = max(0, int((cx - box_width / 2) * width))
        y = max(0, int((cy - box_height / 2) * height))
        right = min(width, int((cx + box_width / 2) * width))
        bottom = min(height, int((cy + box_height / 2) * height))
        rect_width, rect_height = right - x, bottom - y
        if rect_width < 3 or rect_height < 3:
            continue
        grabcut_mask = np.zeros((height, width), dtype=np.uint8)
        background = np.zeros((1, 65), dtype=np.float64)
        foreground = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            image,
            grabcut_mask,
            (x, y, rect_width, rect_height),
            background,
            foreground,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
        foreground_mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        combined = cv2.bitwise_or(combined, foreground_mask)
    if cv2.countNonZero(combined) < 16:
        raise ValueError("Unable to estimate an insulator foreground mask from YOLO boxes")
    return combined


def _draem_region_mask(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Create deterministic smooth anomaly geometry following DRAEM's Perlin-mask principle."""
    height, width = shape
    rng = np.random.default_rng(seed)
    coarse_height = int(rng.choice((2, 4, 8)))
    coarse_width = int(rng.choice((2, 4, 8)))
    coarse = rng.random((coarse_height, coarse_width), dtype=np.float32)
    smooth = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.GaussianBlur(smooth, (0, 0), max(2.0, min(height, width) / 48.0))
    threshold = float(np.quantile(smooth, rng.uniform(0.55, 0.75)))
    return (smooth > threshold).astype(np.uint8) * 255


def _box_mask(
    shape: tuple[int, int],
    box: tuple[int, float, float, float, float],
) -> np.ndarray:
    """Return a conservative pixel mask for one normalized YOLO box."""
    height, width = shape
    _, cx, cy, box_width, box_height = box
    left = max(0, int((cx - box_width / 2) * width))
    top = max(0, int((cy - box_height / 2) * height))
    right = min(width, int((cx + box_width / 2) * width))
    bottom = min(height, int((cy + box_height / 2) * height))
    mask = np.zeros((height, width), dtype=np.uint8)
    if right - left >= 3 and bottom - top >= 3:
        cv2.rectangle(mask, (left, top), (right - 1, bottom - 1), 255, -1)
    return mask


def _object_mask_for_box(
    image: np.ndarray,
    box: tuple[int, float, float, float, float],
) -> np.ndarray:
    """Estimate one object mask and fall back to its documented YOLO box."""
    fallback = _box_mask(image.shape[:2], box)
    try:
        estimated = _grabcut_object_mask(image, [box])
    except (ValueError, cv2.error):
        estimated = fallback
    if cv2.countNonZero(estimated) < 16:
        estimated = fallback
    if cv2.countNonZero(estimated) < 16:
        raise ValueError("Unable to obtain a usable insulator object mask")
    return estimated


def _normalized_mask_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    binary = (mask > 0).astype(np.uint8)
    if cv2.countNonZero(binary) < 16:
        raise ValueError("Synthetic defect mask is too small")
    height, width = binary.shape
    x, y, box_width, box_height = cv2.boundingRect(binary)
    return (
        (x + box_width / 2) / width,
        (y + box_height / 2) / height,
        box_width / width,
        box_height / height,
    )


def _synthesize_broken(
    image: np.ndarray,
    object_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a local foreground section with inpainting to simulate breakage."""
    rng = np.random.default_rng(seed)
    x, y, width, height = cv2.boundingRect((object_mask > 0).astype(np.uint8))
    if width < 6 or height < 6:
        raise ValueError("Insulator foreground is too small for broken synthesis")
    foreground_y, foreground_x = np.where(object_mask > 0)
    selected = int(rng.integers(0, len(foreground_x)))
    center_x = int(foreground_x[selected])
    center_y = int(foreground_y[selected])
    radius_x = max(3, int(width * rng.uniform(0.12, 0.24)))
    radius_y = max(3, int(height * rng.uniform(0.10, 0.20)))
    region = np.zeros_like(object_mask)
    cv2.ellipse(
        region,
        (center_x, center_y),
        (radius_x, radius_y),
        float(rng.uniform(-70.0, 70.0)),
        0,
        360,
        255,
        -1,
    )
    defect_mask = cv2.bitwise_and(region, object_mask)
    if cv2.countNonZero(defect_mask) < 16:
        defect_mask = cv2.bitwise_and(
            cv2.dilate(region, np.ones((7, 7), dtype=np.uint8), iterations=2),
            object_mask,
        )
    if cv2.countNonZero(defect_mask) < 16:
        distances = (foreground_x - center_x) ** 2 + (foreground_y - center_y) ** 2
        nearest = np.argsort(distances)[: min(len(distances), max(16, len(distances) // 8))]
        defect_mask = np.zeros_like(object_mask)
        defect_mask[foreground_y[nearest], foreground_x[nearest]] = 255
        defect_mask = cv2.morphologyEx(
            defect_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
    _normalized_mask_box(defect_mask)
    inpaint_mask = cv2.dilate(defect_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    output = cv2.inpaint(image, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
    rim = cv2.subtract(
        cv2.dilate(defect_mask, np.ones((5, 5), dtype=np.uint8), iterations=1),
        defect_mask,
    )
    rim_alpha = cv2.GaussianBlur((rim > 0).astype(np.float32), (0, 0), 1.2)[..., None]
    output = np.clip(output.astype(np.float32) * (1.0 - 0.22 * rim_alpha), 0, 255).astype(np.uint8)
    return output, defect_mask


def _synthesize_pollution_from_normal(
    image: np.ndarray,
    object_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create deterministic self-texture contamination inside the foreground."""
    rng = np.random.default_rng(seed)
    region = _draem_region_mask(image.shape[:2], seed)
    defect_mask = cv2.bitwise_and(region, object_mask)
    if cv2.countNonZero(defect_mask) < 16:
        defect_mask = cv2.bitwise_and(
            cv2.dilate(region, np.ones((9, 9), dtype=np.uint8), iterations=2),
            object_mask,
        )
    _normalized_mask_box(defect_mask)
    alpha = float(rng.uniform(0.38, 0.68))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    low_frequency = cv2.GaussianBlur(lab, (0, 0), max(3.0, min(image.shape[:2]) / 24.0))
    texture = low_frequency.copy()
    texture[..., 0] = np.clip(texture[..., 0] - rng.uniform(28.0, 60.0), 0, 255)
    texture[..., 1] = np.clip(texture[..., 1] + rng.uniform(4.0, 14.0), 0, 255)
    texture[..., 2] = np.clip(texture[..., 2] + rng.uniform(8.0, 22.0), 0, 255)
    feather = cv2.GaussianBlur((defect_mask > 0).astype(np.float32), (0, 0), 2.5)
    feather = np.clip(feather * alpha, 0.0, 1.0)[..., None]
    blended = lab * (1.0 - feather) + texture * feather
    output = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return output, defect_mask, alpha


def add_python_defect_synthesis(
    output_root: Path,
    broken_per_split: int,
    pollution_per_split: int,
    seed: int = 42,
) -> int:
    """Generate traceable broken and pollution samples from normal split-local images.

    Validation output is deliberately identified as synthetic validation. It is
    useful for regression and smoke evaluation, not evidence of field accuracy.
    """
    if broken_per_split < 1 or pollution_per_split < 1:
        raise ValueError("broken_per_split and pollution_per_split must be positive")
    output_root = output_root.resolve()
    manifest_file = output_root / "sources.jsonl"
    manifest = [
        json.loads(line)
        for line in manifest_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base_by_split: dict[str, list[dict]] = {}
    for split in ("train", "val"):
        prefix = f"images/{split}/"
        base_by_split[split] = [
            record
            for record in manifest
            if record.get("origin") == "source-copy"
            and "normal" in record.get("categories", [])
            and str(record.get("image", "")).startswith(prefix)
        ]
        if not base_by_split[split]:
            raise ValueError(f"No traceable normal {split} images are available for synthesis")
    train_hashes = {record["image_sha256"] for record in base_by_split["train"]}
    val_hashes = {record["image_sha256"] for record in base_by_split["val"]}
    overlap = train_hashes & val_hashes
    if overlap:
        raise ValueError("Normal source images overlap between train and validation")

    generated = 0
    object_mask_cache: dict[tuple[str, int], np.ndarray] = {}
    specifications = (
        ("broken", broken_per_split, _synthesize_broken),
        ("pollution", pollution_per_split, _synthesize_pollution_from_normal),
    )
    for split_index, split in enumerate(("train", "val")):
        bases = base_by_split[split]
        for category_index, (category, count, synthesizer) in enumerate(specifications):
            for index in range(count):
                base_record = bases[(index + category_index * count) % len(bases)]
                image_path = output_root / base_record["image"]
                label_path = output_root / "labels" / split / f"{image_path.stem}.txt"
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Unable to decode normal source image {image_path}")
                labels = parse_yolo_labels(label_path)
                normal_indices = [i for i, box in enumerate(labels) if box[0] == CLASS_IDS["normal"]]
                if not normal_indices:
                    raise ValueError(f"No normal object label in {label_path}")
                sample_seed = seed + split_index * 1_000_000 + category_index * 100_000 + index
                selected_index = normal_indices[sample_seed % len(normal_indices)]
                cache_key = (str(image_path), selected_index)
                if cache_key not in object_mask_cache:
                    object_mask_cache[cache_key] = _object_mask_for_box(
                        image,
                        labels[selected_index],
                    )
                object_mask = object_mask_cache[cache_key]
                synthetic = synthesizer(image, object_mask, sample_seed)
                if category == "pollution":
                    output, defect_mask, alpha = synthetic
                else:
                    output, defect_mask = synthetic
                    alpha = None
                defect_box = _normalized_mask_box(defect_mask)
                output_labels = list(labels)
                output_labels[selected_index] = (CLASS_IDS[category], *defect_box)
                stem = f"{category}_python_{split}_{index:06d}"
                target_image = output_root / "images" / split / f"{stem}.jpg"
                target_label = output_root / "labels" / split / f"{stem}.txt"
                if target_image.exists() or target_label.exists():
                    raise ValueError(f"Duplicate target sample {stem}")
                if not cv2.imwrite(str(target_image), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise ValueError(f"Unable to write {target_image}")
                write_yolo_labels(target_label, output_labels)
                record = {
                    "image": str(target_image.relative_to(output_root)),
                    "source_dataset": "CPLID+InsuLens-Python-synthesis",
                    "source_url": CPLID_SOURCE_URL,
                    "source_commit": CPLID_COMMIT,
                    "origin": f"python-{category}-from-normal",
                    "pixel_transform": (
                        "GrabCut foreground + local inpainting removal"
                        if category == "broken"
                        else "GrabCut foreground + DRAEM smooth mask + self-texture LAB contamination"
                    ),
                    "synthetic": True,
                    "synthetic_validation": split == "val",
                    "validation_domain": "synthetic" if split == "val" else "training",
                    "field_validation_eligible": False,
                    "random_seed": sample_seed,
                    "categories": sorted({TRAIN_CLASS_NAMES[label[0]] for label in output_labels}),
                    "synthetic_category": category,
                    "source_image": str(image_path.resolve()),
                    "source_label": str(label_path.resolve()),
                    "input_sha256": {
                        "image": _sha256(image_path),
                        "label": _sha256(label_path),
                    },
                }
                if alpha is not None:
                    record["alpha"] = alpha
                manifest.append(record)
                generated += 1
    manifest_file.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest),
        encoding="utf-8",
    )
    _write_metadata(output_root, manifest)
    return generated


def add_dtd_pollution_synthesis(
    output_root: Path,
    dtd_root: Path,
    count: int,
    seed: int = 42,
) -> int:
    """Add train-only DRAEM-style pollution using DTD textures and CPLID boxes.

    DTD is restricted to research use by its dataset page. Synthetic samples
    never satisfy validation coverage; real pollution validation data remains
    mandatory before training.
    """
    if count < 1:
        raise ValueError("count must be positive")
    texture_paths = sorted(
        path
        for category in DTD_POLLUTION_TEXTURE_CLASSES
        for path in (dtd_root / "images" / category).glob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not texture_paths:
        raise ValueError(f"No pollution-relevant DTD textures found in {dtd_root}")
    manifest_file = output_root / "sources.jsonl"
    manifest = [json.loads(line) for line in manifest_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    base_records = [
        record
        for record in manifest
        if record.get("origin") == "source-copy"
        and "normal" in record.get("categories", [])
        and str(record.get("image", "")).startswith("images/train/")
    ]
    if not base_records:
        raise ValueError("No traceable normal train images are available for pollution synthesis")
    generated = 0
    for index in range(count):
        base_record = base_records[index % len(base_records)]
        image_path = output_root / base_record["image"]
        label_path = output_root / "labels" / "train" / f"{image_path.stem}.txt"
        texture_path = texture_paths[index % len(texture_paths)]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        texture = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
        if image is None or texture is None:
            raise ValueError("Unable to decode DTD pollution synthesis inputs")
        object_mask = _grabcut_object_mask(image, parse_yolo_labels(label_path))
        region_mask = _draem_region_mask(image.shape[:2], seed + index)
        blend_mask = cv2.bitwise_and(object_mask, region_mask)
        if cv2.countNonZero(blend_mask) < 16:
            region_mask = cv2.dilate(region_mask, np.ones((9, 9), dtype=np.uint8), iterations=2)
            blend_mask = cv2.bitwise_and(object_mask, region_mask)
        if cv2.countNonZero(blend_mask) < 16:
            raise ValueError(f"DRAEM mask does not overlap the insulator for sample {index}")
        height, width = image.shape[:2]
        texture = cv2.resize(texture, (width, height), interpolation=cv2.INTER_LINEAR)
        feather = cv2.GaussianBlur((blend_mask > 0).astype(np.float32), (0, 0), 2.0)
        alpha = float(np.random.default_rng(seed + index).uniform(0.35, 0.65))
        feather = np.clip(feather * alpha, 0.0, 1.0)[..., None]
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_texture = cv2.cvtColor(texture, cv2.COLOR_BGR2LAB).astype(np.float32)
        blended = lab_image * (1.0 - feather) + lab_texture * feather
        output = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        stem = f"pollution_dtd_{index:06d}"
        target_image = output_root / "images" / "train" / f"{stem}.jpg"
        target_label = output_root / "labels" / "train" / f"{stem}.txt"
        if target_image.exists() or target_label.exists():
            raise ValueError(f"Duplicate target sample {stem}")
        if not cv2.imwrite(str(target_image), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise ValueError(f"Unable to write {target_image}")
        x, y, box_width, box_height = cv2.boundingRect((blend_mask > 0).astype(np.uint8))
        box = (
            (x + box_width / 2) / width,
            (y + box_height / 2) / height,
            box_width / width,
            box_height / height,
        )
        write_yolo_labels(target_label, [(CLASS_IDS["pollution"], *box)])
        manifest.append(
            {
                "image": str(target_image.relative_to(output_root)),
                "source_dataset": "CPLID+DRAEM+DTD",
                "source_url": DTD_SOURCE_URL,
                "license": DTD_LICENSE,
                "origin": "DRAEM-DTD-GrabCut-synthesis",
                "method_reference": DRAEM_SOURCE_URL,
                "pixel_transform": "GrabCut foreground + smooth anomaly mask + LAB texture blend",
                "validation_eligible": False,
                "random_seed": seed + index,
                "alpha": alpha,
                "categories": ["pollution"],
                "source_image": str(image_path.resolve()),
                "source_label": str(label_path.resolve()),
                "texture": str(texture_path.resolve()),
                "input_sha256": {
                    "image": _sha256(image_path),
                    "label": _sha256(label_path),
                    "texture": _sha256(texture_path),
                },
            }
        )
        generated += 1
    manifest_file.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest), encoding="utf-8")
    _write_metadata(output_root, manifest)
    return generated


def add_pollution_synthesis(
    output_root: Path,
    manifest_path: Path,
    split: str = "train",
) -> int:
    """Add pollution images described by a provenance JSONL manifest."""
    if split != "train":
        raise ValueError("Synthetic pollution is restricted to train; validation must be real")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_file = output_root / "sources.jsonl"
    manifest = [json.loads(line) for line in manifest_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, record in enumerate(records):
        required = ("image", "object_mask", "texture", "texture_mask", "source_url", "license")
        missing = [key for key in required if not record.get(key)]
        if missing:
            raise ValueError(f"Pollution manifest record {index} is missing {missing}")
        inputs = {key: Path(record[key]).expanduser().resolve() for key in required[:4]}
        stem = f"pollution_synthetic_{index:06d}"
        target_image = output_root / "images" / split / f"{stem}.jpg"
        target_label = output_root / "labels" / split / f"{stem}.txt"
        box = synthesize_pollution_sample(**{f"{key}_path": value for key, value in inputs.items()}, output_path=target_image, alpha=float(record.get("alpha", 0.55)))
        write_yolo_labels(target_label, [(CLASS_IDS["pollution"], *box)])
        manifest.append(
            {
                "image": str(target_image.relative_to(output_root)),
                "source_dataset": record.get("source_dataset", "user-provided-pollution-texture"),
                "source_url": record["source_url"],
                "license": record["license"],
                "origin": "DRAEM-style-real-texture-mask-blend",
                "method_reference": DRAEM_SOURCE_URL,
                "pixel_transform": "LAB masked blend",
                "alpha": float(record.get("alpha", 0.55)),
                "categories": ["pollution"],
                "inputs": {key: str(value) for key, value in inputs.items()},
                "input_sha256": {key: _sha256(value) for key, value in inputs.items()},
            }
        )
    manifest_file.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest), encoding="utf-8")
    _write_metadata(output_root, manifest)
    return len(records)


def _write_metadata(output_root: Path, manifest: list[dict]) -> dict:
    label_counts = {name: 0 for name in TRAIN_CLASS_NAMES}
    image_counts = {name: 0 for name in TRAIN_CLASS_NAMES}
    split_label_counts = {
        split: {name: 0 for name in TRAIN_CLASS_NAMES}
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        for _, label_path in _source_pairs(output_root / "images" / split, output_root / "labels" / split):
            categories = set()
            for class_id, *_ in parse_yolo_labels(label_path):
                if class_id not in range(len(TRAIN_CLASS_NAMES)):
                    raise ValueError(f"{label_path} uses unsupported class {class_id}")
                category = TRAIN_CLASS_NAMES[class_id]
                label_counts[category] += 1
                split_label_counts[split][category] += 1
                categories.add(category)
            for category in categories:
                image_counts[category] += 1
    metadata = {
        "schema": "insulator-four-trained-five-report-yolo-v3",
        "classes": list(TRAIN_CLASS_NAMES),
        "report_classes": list(REPORT_CLASS_NAMES),
        "report_only_classes": ["flashover"],
        "label_counts": label_counts,
        "image_counts": image_counts,
        "split_label_counts": split_label_counts,
        "unavailable_classes": [name for name, count in label_counts.items() if count == 0],
        "validation_unavailable_classes": [
            name for name, count in split_label_counts["val"].items() if count == 0
        ],
        "synthetic_validation_records": sum(
            bool(record.get("synthetic_validation")) for record in manifest
        ),
        "field_validation_ready": all(
            any(
                str(record.get("image", "")).startswith("images/val/")
                and name in record.get("categories", [])
                and not record.get("synthetic")
                for record in manifest
            )
            for name in TRAIN_CLASS_NAMES
        ),
        "source_records": len(manifest),
        "pixel_policy": "real images are copied; synthetic defects retain source hashes, parameters, and seeds",
        "training_gate": "all classes require split-local traceable records; synthetic validation is not field validation",
    }
    (output_root / "dataset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_insulator_dataset(
    cplid_root: Path,
    output_root: Path,
    broken_root: Path | None = None,
    pollution_root: Path | None = None,
    per_split: int = 80,
) -> dict:
    """Build CPLID plus optional real broken and pollution YOLO sources."""
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _reset_output(output_root)
    manifest: list[dict] = []
    _copy_dataset(cplid_root.resolve(), output_root, "CPLID", CPLID_SOURCE_URL, {0: CLASS_IDS["normal"], 1: CLASS_IDS["missing"]}, manifest, per_split)
    if broken_root:
        # TLID documents class 0 as damage. Callers with another source must
        # normalize it to one damage class before import.
        _copy_dataset(broken_root.resolve(), output_root, "TLID", TLID_SOURCE_URL, {0: CLASS_IDS["broken"]}, manifest)
    if pollution_root:
        _copy_dataset(pollution_root.resolve(), output_root, "REAL_POLLUTION", "user-provided-provenance", {0: CLASS_IDS["pollution"]}, manifest)
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(output_root), "train": "images/train", "val": "images/val", "names": {index: name for index, name in enumerate(TRAIN_CLASS_NAMES)}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output_root / "sources.jsonl").write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest), encoding="utf-8")
    return _write_metadata(output_root, manifest)


def validate_insulator_dataset(root: Path, require_all_classes: bool = False) -> dict:
    config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = config.get("names", {})
    ordered = tuple(names[index] if index in names else names[str(index)] for index in range(len(TRAIN_CLASS_NAMES)))
    if ordered != TRAIN_CLASS_NAMES:
        raise ValueError(f"Expected classes {TRAIN_CLASS_NAMES}, got {ordered}")
    manifest_path = root / "sources.jsonl"
    if not manifest_path.is_file():
        raise ValueError("Training blocked: sources.jsonl is missing")
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_image = {entry["image"]: entry for entry in manifest}
    label_counts = {name: 0 for name in TRAIN_CLASS_NAMES}
    split_label_counts = {
        split: {name: 0 for name in TRAIN_CLASS_NAMES}
        for split in ("train", "val")
    }
    image_count = 0
    for split in ("train", "val"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        pairs = _source_pairs(image_dir, label_dir)
        image_files = {path.stem for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
        label_files = {path.stem for path in label_dir.glob("*.txt")}
        if image_files != label_files:
            raise ValueError(f"Unpaired images or labels in {root}/{split}")
        image_count += len(pairs)
        for image_path, label_path in pairs:
            relative = str(image_path.relative_to(root))
            if relative not in by_image:
                raise ValueError(f"Training blocked: no provenance for {relative}")
            for class_id, *_ in parse_yolo_labels(label_path):
                if class_id not in range(len(TRAIN_CLASS_NAMES)):
                    raise ValueError(f"{label_path} uses unsupported class {class_id}")
                category = TRAIN_CLASS_NAMES[class_id]
                label_counts[category] += 1
                split_label_counts[split][category] += 1
    missing_classes = [name for name, count in label_counts.items() if count == 0]
    missing_train_classes = [name for name, count in split_label_counts["train"].items() if count == 0]
    missing_validation_classes = [name for name, count in split_label_counts["val"].items() if count == 0]
    blocked_classes = [
        name
        for name in TRAIN_CLASS_NAMES
        if name in missing_train_classes or name in missing_validation_classes
    ]
    if require_all_classes and blocked_classes:
        raise ValueError(
            "Training blocked: no traceable train/validation samples for classes "
            + ", ".join(blocked_classes)
        )
    return {
        "classes": list(ordered),
        "report_classes": list(REPORT_CLASS_NAMES),
        "images": image_count,
        "labels": sum(label_counts.values()),
        "label_counts": label_counts,
        "split_label_counts": split_label_counts,
        "missing_classes": missing_classes,
        "missing_train_classes": missing_train_classes,
        "missing_validation_classes": missing_validation_classes,
        "training_ready": not blocked_classes,
        "synthetic_validation": any(
            entry.get("synthetic_validation") for entry in manifest
        ),
        "field_validation_ready": all(
            any(
                str(entry.get("image", "")).startswith("images/val/")
                and name in entry.get("categories", [])
                and not entry.get("synthetic")
                for entry in manifest
            )
            for name in TRAIN_CLASS_NAMES
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the traceable InsuLens dataset")
    parser.add_argument("--cplid", type=Path, default=Path("datasets/cplid_yolo"))
    parser.add_argument("--broken", type=Path)
    parser.add_argument("--pollution", type=Path)
    parser.add_argument("--pollution-synthesis-manifest", type=Path)
    parser.add_argument("--dtd-textures", type=Path)
    parser.add_argument("--dtd-pollution-count", type=int, default=0)
    parser.add_argument("--python-broken-per-split", type=int, default=0)
    parser.add_argument("--python-pollution-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("datasets/insulator_five_report"))
    parser.add_argument("--per-split", type=int, default=80)
    args = parser.parse_args()
    metadata = build_insulator_dataset(args.cplid, args.output, args.broken, args.pollution, args.per_split)
    if args.pollution_synthesis_manifest:
        add_pollution_synthesis(args.output.resolve(), args.pollution_synthesis_manifest.resolve())
        metadata = json.loads((args.output / "dataset_metadata.json").read_text(encoding="utf-8"))
    if args.dtd_pollution_count:
        if not args.dtd_textures:
            parser.error("--dtd-textures is required with --dtd-pollution-count")
        add_dtd_pollution_synthesis(
            args.output.resolve(),
            args.dtd_textures.resolve(),
            args.dtd_pollution_count,
            args.seed,
        )
        metadata = json.loads((args.output / "dataset_metadata.json").read_text(encoding="utf-8"))
    if args.python_broken_per_split or args.python_pollution_per_split:
        if not args.python_broken_per_split or not args.python_pollution_per_split:
            parser.error(
                "--python-broken-per-split and --python-pollution-per-split must be used together"
            )
        add_python_defect_synthesis(
            args.output.resolve(),
            args.python_broken_per_split,
            args.python_pollution_per_split,
            args.seed,
        )
        metadata = json.loads((args.output / "dataset_metadata.json").read_text(encoding="utf-8"))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(validate_insulator_dataset(args.output.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
