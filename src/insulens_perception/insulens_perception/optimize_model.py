"""Structured pruning, fine-tuning and deployment export for InsuLens YOLO weights.

Pruning is intentionally explicit rather than silently altering a model during
ordinary training.  First produce a baseline weight, then prune and fine-tune
against the same validation split before accepting the compressed artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    options = argparse.ArgumentParser(description="Prune and export InsuLens defect detector")
    options.add_argument("--weights", required=True, help="Baseline four-trained-class .pt weight")
    options.add_argument("--data", required=True, help="YOLO data.yaml used for recovery fine-tuning")
    options.add_argument("--output", default="models/insulator_defect_pruned.pt")
    options.add_argument("--sparsity", type=float, default=0.2, help="Structured channel pruning ratio [0, 0.8)")
    options.add_argument("--epochs", type=int, default=25, help="Recovery fine-tuning epochs")
    options.add_argument("--imgsz", type=int, default=960)
    options.add_argument("--device", default="0")
    options.add_argument("--format", choices=("onnx", "engine", "openvino"), default="onnx")
    options.add_argument("--int8", action="store_true", help="Request calibrated INT8 export when backend supports it")
    return options


def main() -> None:
    args = parser().parse_args()
    if not 0 <= args.sparsity < 0.8:
        raise SystemExit("--sparsity 必须在 [0, 0.8) 内。")
    try:
        import torch
        from torch.nn.utils import prune
        from .modeling import create_yolo
    except ImportError as error:
        raise SystemExit(f"缺少剪枝依赖：{error}") from error
    model = create_yolo(str(Path(args.weights).expanduser().resolve()))
    modules = [module for module in model.model.modules() if isinstance(module, torch.nn.Conv2d) and module.groups == 1]
    for module in modules:
        prune.ln_structured(module, name="weight", amount=args.sparsity, n=2, dim=0)
        prune.remove(module, "weight")
    # Recovery training preserves the baseline's architecture/classes while
    # adapting the zeroed structured channels to the actual fault data.
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, device=args.device, lr0=1e-4, cos_lr=True)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), output.with_suffix(".state_dict.pt"))
    exported = model.export(format=args.format, imgsz=args.imgsz, int8=args.int8, simplify=True)
    print(f"Pruned state dict: {output.with_suffix('.state_dict.pt')}")
    print(f"Deployment artifact: {exported}")
    print("请用 model.val(data=...) 对比基线 mAP、召回率、延迟与模型大小后再部署。")


if __name__ == "__main__":
    main()
