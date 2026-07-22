"""Train YOLOv10s on the Gazebo-generated insulator dataset."""

import argparse
from pathlib import Path
import shutil


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv10 on the synthetic insulator dataset"
    )
    parser.add_argument(
        "--data", default="/root/doggo/datasets/insulator_sim/data.yaml"
    )
    parser.add_argument("--model", default="yolov10s.yaml")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="/root/doggo/runs")
    parser.add_argument("--name", default="insulator_yolov10s")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--export", default="/root/doggo/models/insulator_yolov10s.pt"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(
            f"Dataset config does not exist: {data_path}. "
            "Run generate_dataset.launch.py first."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run scripts/setup_env.sh first."
        ) from exc

    model = YOLO(args.model)
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
    best_weight = save_dir / "weights" / "best.pt"
    export_path = Path(args.export).expanduser().resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, export_path)
    print(f"Training complete: {best_weight}")
    print(f"Runtime weight exported to: {export_path}")


if __name__ == "__main__":
    main()
