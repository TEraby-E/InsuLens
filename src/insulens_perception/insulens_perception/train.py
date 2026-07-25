"""Train YOLOv10s on four observable classes used by five-class reporting."""

import argparse
import json
from pathlib import Path
import shutil

from .modeling import create_yolo
from .insulator_dataset import TRAIN_CLASS_NAMES, validate_insulator_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv10 on the four trained insulator classes"
    )
    parser.add_argument(
        "--data", default="datasets/insulator_five_report/data.yaml"
    )
    parser.add_argument("--model", default="yolov10s.yaml")
    parser.add_argument(
        "--small-object-model",
        action="store_true",
        help="Train the version-controlled P2 + Coordinate Attention model.",
    )
    parser.add_argument(
        "--tal-topk",
        type=int,
        choices=(6, 10, 13),
        default=10,
        help="Record the Task-Aligned Assigner ablation candidate.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="/root/insulens/runs")
    parser.add_argument("--name", default="insulator_five_report_yolov10s")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--export", default="models/insulator_five_report_yolov10s.pt"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(
            f"Dataset config does not exist: {data_path}. "
            "Run python -m insulens_perception.insulator_dataset first."
        )
    validated = validate_insulator_dataset(data_path.parent, require_all_classes=True)

    model_path = args.model
    if args.small_object_model:
        model_path = str(
            Path(__file__).resolve().parents[1] / "models" / "yolov10s_p2_ca.yaml"
        )
    try:
        model = create_yolo(model_path)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).expanduser().resolve()),
        name=args.name,
        patience=args.patience,
        pretrained=args.model.endswith(".pt"),
        amp=False,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        degrees=12.0,
        translate=0.15,
        scale=0.45,
        fliplr=0.5,
        mixup=0.05,
        exist_ok=True,
    )
    save_dir = Path(result.save_dir)
    (save_dir / "small_object_experiment.json").write_text(
        json.dumps(
            {
                "model": model_path,
                "p2_coordinate_attention": args.small_object_model,
                "tal_topk_candidate": args.tal_topk,
                "classes": list(TRAIN_CLASS_NAMES),
                "dataset_validation": validated,
                "note": "Record TAL candidates for ablation; the installed Ultralytics release owns its assigner implementation.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    best_weight = save_dir / "weights" / "best.pt"
    export_path = Path(args.export).expanduser().resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, export_path)
    print(f"Training complete: {best_weight}")
    print(f"Runtime weight exported to: {export_path}")


if __name__ == "__main__":
    main()
