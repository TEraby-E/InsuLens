#!/usr/bin/env python3
"""Convert CPLID VOC annotations into a two-class YOLO dataset.

Class 0 is the complete insulator string and class 1 is the localized
missing-disc defect supplied as ``defect`` by CPLID.  Normal and defective
source groups are split independently to retain both groups in validation.
"""

import argparse
import json
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET


CLASS_IDS = {"insulator": 0, "defect": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=(
            "/root/InsuLens/datasets/raw/cplid/"
            "InsulatorDataSet-master"
        ),
    )
    parser.add_argument(
        "--output", default="/root/InsuLens/datasets/cplid_yolo"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    labels = []
    for item in root.findall("object"):
        name = item.findtext("name")
        if name not in CLASS_IDS:
            continue
        box = item.find("bndbox")
        xmin = max(0.0, float(box.findtext("xmin")))
        ymin = max(0.0, float(box.findtext("ymin")))
        xmax = min(width, float(box.findtext("xmax")))
        ymax = min(height, float(box.findtext("ymax")))
        if xmax <= xmin or ymax <= ymin:
            continue
        labels.append(
            (
                CLASS_IDS[name],
                ((xmin + xmax) / 2.0) / width,
                ((ymin + ymax) / 2.0) / height,
                (xmax - xmin) / width,
                (ymax - ymin) / height,
            )
        )
    return labels


def stratified_split(items, val_ratio: float, rng: random.Random):
    items = list(items)
    rng.shuffle(items)
    val_count = max(1, round(len(items) * val_ratio))
    return items[val_count:], items[:val_count]


def prepare_item(kind: str, image_path: Path, source: Path):
    stem = image_path.stem
    if kind == "normal":
        label_files = [source / "Normal_Insulators" / "labels" / f"{stem}.xml"]
    else:
        label_root = source / "Defective_Insulators" / "labels"
        label_files = [
            label_root / "insulator" / f"{stem}.xml",
            label_root / "defect" / f"{stem}.xml",
        ]
    missing = [path for path in label_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing annotations for {image_path}: {missing}")
    return kind, image_path, label_files


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1")
    if not source.is_dir():
        raise SystemExit(
            f"CPLID source not found: {source}. Run scripts/download_cplid.sh."
        )

    rng = random.Random(args.seed)
    normal = [
        prepare_item("normal", path, source)
        for path in sorted((source / "Normal_Insulators" / "images").glob("*.jpg"))
    ]
    defective = [
        prepare_item("defective", path, source)
        for path in sorted(
            (source / "Defective_Insulators" / "images").glob("*.jpg")
        )
    ]
    normal_train, normal_val = stratified_split(normal, args.val_ratio, rng)
    defect_train, defect_val = stratified_split(defective, args.val_ratio, rng)
    splits = {
        "train": normal_train + defect_train,
        "val": normal_val + defect_val,
    }

    for split, records in splits.items():
        image_output = output / "images" / split
        label_output = output / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        expected = set()
        for kind, image_path, xml_paths in records:
            output_stem = f"cplid_{kind}_{image_path.stem}"
            expected.add(output_stem)
            shutil.copy2(image_path, image_output / f"{output_stem}.jpg")
            labels = []
            for xml_path in xml_paths:
                labels.extend(read_voc(xml_path))
            text = "".join(
                f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n"
                for class_id, cx, cy, width, height in labels
            )
            (label_output / f"{output_stem}.txt").write_text(
                text, encoding="utf-8"
            )

        for directory, suffix in ((image_output, ".jpg"), (label_output, ".txt")):
            for stale in directory.glob(f"*{suffix}"):
                if stale.stem not in expected:
                    stale.unlink()

    data_yaml = (
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: insulator\n"
        "  1: missing_disc\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    metadata = {
        "dataset": "Chinese Power Line Insulator Dataset (CPLID)",
        "source": "https://github.com/InsulatorData/InsulatorDataSet",
        "commit": "1f6349f619237344d49905090ecf2704505394a4",
        "citation": "Tao et al., IEEE TSMC: Systems, 2018",
        "license_note": (
            "The upstream repository does not declare an explicit data license. "
            "Use for research/coursework with citation; do not redistribute."
        ),
        "defect_note": (
            "Normal images are UAV captures. Defective images use synthesized "
            "defective insulators on transmission-scene backgrounds."
        ),
        "class_mapping": {"insulator": 0, "defect": "missing_disc (1)"},
        "counts": {
            "normal_images": len(normal),
            "defective_images": len(defective),
            "train_images": len(splits["train"]),
            "val_images": len(splits["val"]),
        },
        "seed": args.seed,
    }
    (output / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"CPLID converted: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, output={output}"
    )


if __name__ == "__main__":
    main()
