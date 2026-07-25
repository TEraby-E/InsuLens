import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from insulens_perception.insulator_dataset import (
    REPORT_CLASS_NAMES,
    TRAIN_CLASS_NAMES,
    add_dtd_pollution_synthesis,
    add_pollution_synthesis,
    add_python_defect_synthesis,
    build_insulator_dataset,
    validate_insulator_dataset,
)


def _write_source_sample(root, split: str, stem: str, class_id: int, value: int = 150) -> None:
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((96, 128, 3), value, dtype=np.uint8)
    cv2.rectangle(image, (45, 15), (80, 82), (240, 240, 240), -1)
    assert cv2.imwrite(str(image_dir / f"{stem}.jpg"), image)
    (label_dir / f"{stem}.txt").write_text(
        f"{class_id} 0.49 0.51 0.28 0.70\n", encoding="utf-8"
    )


def _write_cplid(root) -> None:
    for split_index, split in enumerate(("train", "val")):
        _write_source_sample(root, split, f"normal_{split}", 0, 150 + split_index)
        _write_source_sample(root, split, f"missing_{split}", 1, 150 + split_index)


def test_builder_copies_cplid_and_reports_missing_defect_sources(tmp_path):
    source = tmp_path / "cplid"
    _write_cplid(source)
    output = tmp_path / "insulator"

    metadata = build_insulator_dataset(source, output, per_split=2)
    validation = validate_insulator_dataset(output)
    records = [json.loads(line) for line in (output / "sources.jsonl").read_text().splitlines()]

    assert metadata["classes"] == list(TRAIN_CLASS_NAMES)
    assert metadata["report_classes"] == list(REPORT_CLASS_NAMES)
    assert metadata["unavailable_classes"] == ["broken", "pollution"]
    assert validation["missing_classes"] == ["broken", "pollution"]
    assert all(record["pixel_transform"] == "none" for record in records)
    for record in records:
        assert (output / record["image"]).read_bytes() == Path(record["source_image"]).read_bytes()


def test_tlid_damage_and_real_pollution_make_dataset_training_ready(tmp_path):
    cplid, broken, pollution = tmp_path / "cplid", tmp_path / "tlid", tmp_path / "pollution"
    _write_cplid(cplid)
    for split in ("train", "val"):
        _write_source_sample(broken, split, f"damage_{split}", 0, 120)
        _write_source_sample(pollution, split, f"pollution_{split}", 0, 90)
    output = tmp_path / "insulator"

    metadata = build_insulator_dataset(cplid, output, broken, pollution, per_split=2)
    validation = validate_insulator_dataset(output, require_all_classes=True)

    assert metadata["unavailable_classes"] == []
    assert validation["training_ready"] is True
    assert validation["label_counts"] == {
        "normal": 2,
        "broken": 2,
        "pollution": 2,
        "missing": 2,
    }
    records = [json.loads(line) for line in (output / "sources.jsonl").read_text().splitlines()]
    broken_records = [record for record in records if record["categories"] == ["broken"]]
    assert broken_records and all(record["source_dataset"] == "TLID" for record in broken_records)


def test_pollution_synthesis_requires_real_texture_masks_and_is_train_only(tmp_path):
    cplid = tmp_path / "cplid"
    _write_cplid(cplid)
    output = tmp_path / "insulator"
    build_insulator_dataset(cplid, output, per_split=2)

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    base = np.full((96, 128, 3), 180, dtype=np.uint8)
    texture = np.full((96, 128, 3), (35, 80, 120), dtype=np.uint8)
    mask = np.zeros((96, 128), dtype=np.uint8)
    cv2.rectangle(mask, (35, 20), (90, 75), 255, -1)
    for name, image in (("base.jpg", base), ("texture.jpg", texture), ("object.png", mask), ("texture_mask.png", mask)):
        assert cv2.imwrite(str(inputs / name), image)
    synthesis_manifest = tmp_path / "pollution.jsonl"
    synthesis_manifest.write_text(
        json.dumps(
            {
                "image": str(inputs / "base.jpg"),
                "object_mask": str(inputs / "object.png"),
                "texture": str(inputs / "texture.jpg"),
                "texture_mask": str(inputs / "texture_mask.png"),
                "source_url": "https://example.invalid/controlled-pollution-capture",
                "license": "internal-controlled-capture",
                "alpha": 0.6,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert add_pollution_synthesis(output, synthesis_manifest) == 1
    synthetic = output / "images" / "train" / "pollution_synthetic_000000.jpg"
    assert synthetic.is_file() and synthetic.read_bytes() != (inputs / "base.jpg").read_bytes()
    record = json.loads((output / "sources.jsonl").read_text().splitlines()[-1])
    assert record["origin"] == "DRAEM-style-real-texture-mask-blend"
    assert set(record["input_sha256"]) == {"image", "object_mask", "texture", "texture_mask"}
    with pytest.raises(ValueError, match="restricted to train"):
        add_pollution_synthesis(output, synthesis_manifest, split="val")


def test_dtd_pollution_synthesis_uses_cplid_boxes_and_keeps_real_val_gate(tmp_path):
    cplid = tmp_path / "cplid"
    _write_cplid(cplid)
    output = tmp_path / "insulator"
    build_insulator_dataset(cplid, output, per_split=2)
    dtd = tmp_path / "dtd"
    texture_dir = dtd / "images" / "stained"
    texture_dir.mkdir(parents=True)
    texture = np.full((96, 128, 3), (30, 75, 125), dtype=np.uint8)
    cv2.circle(texture, (64, 48), 30, (70, 110, 150), -1)
    assert cv2.imwrite(str(texture_dir / "stained_0001.jpg"), texture)

    assert add_dtd_pollution_synthesis(output, dtd, count=1, seed=7) == 1
    validation = validate_insulator_dataset(output)
    record = json.loads((output / "sources.jsonl").read_text().splitlines()[-1])

    assert validation["split_label_counts"]["train"]["pollution"] == 1
    assert validation["split_label_counts"]["val"]["pollution"] == 0
    assert validation["training_ready"] is False
    assert record["origin"] == "DRAEM-DTD-GrabCut-synthesis"
    assert record["validation_eligible"] is False
    assert record["random_seed"] == 7


def test_training_gate_rejects_missing_traceable_classes(tmp_path):
    cplid = tmp_path / "cplid"
    _write_cplid(cplid)
    output = tmp_path / "insulator"
    build_insulator_dataset(cplid, output, per_split=2)

    with pytest.raises(ValueError, match="broken, pollution"):
        validate_insulator_dataset(output, require_all_classes=True)


def test_python_synthesis_generates_split_local_broken_and_pollution(tmp_path):
    cplid = tmp_path / "cplid"
    _write_cplid(cplid)
    output = tmp_path / "insulator"
    build_insulator_dataset(cplid, output, per_split=2)

    assert add_python_defect_synthesis(
        output,
        broken_per_split=1,
        pollution_per_split=1,
        seed=17,
    ) == 4
    validation = validate_insulator_dataset(output, require_all_classes=True)
    records = [json.loads(line) for line in (output / "sources.jsonl").read_text().splitlines()]
    synthetic = [record for record in records if record.get("synthetic")]

    assert validation["training_ready"] is True
    assert validation["synthetic_validation"] is True
    assert validation["field_validation_ready"] is False
    assert validation["split_label_counts"]["train"]["broken"] == 1
    assert validation["split_label_counts"]["train"]["pollution"] == 1
    assert validation["split_label_counts"]["val"]["broken"] == 1
    assert validation["split_label_counts"]["val"]["pollution"] == 1
    assert len(synthetic) == 4
    assert all(record["field_validation_eligible"] is False for record in synthetic)
    assert all(set(record["input_sha256"]) == {"image", "label"} for record in synthetic)
    train_sources = {
        record["input_sha256"]["image"]
        for record in synthetic
        if record["validation_domain"] == "training"
    }
    val_sources = {
        record["input_sha256"]["image"]
        for record in synthetic
        if record["validation_domain"] == "synthetic"
    }
    assert train_sources.isdisjoint(val_sources)
