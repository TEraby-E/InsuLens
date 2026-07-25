"""Version-controlled custom model components for InsuLens experiments."""

from __future__ import annotations

import torch
from torch import nn


class HSwish(nn.Module):
    """Hard-swish activation used by Coordinate Attention."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.nn.functional.relu6(value + 3.0, inplace=True) / 6.0


class CoordAtt(nn.Module):
    """Position-aware channel attention that preserves the input tensor shape."""

    def __init__(self, reduction: int = 32) -> None:
        super().__init__()
        self.reduction = reduction
        self.conv1: nn.Conv2d | None = None
        self.bn1: nn.BatchNorm2d | None = None
        self.conv_h: nn.Conv2d | None = None
        self.conv_w: nn.Conv2d | None = None
        self.activation = HSwish()

    def _build(self, channels: int, value: torch.Tensor) -> None:
        hidden = max(8, channels // self.reduction)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)
        self.to(device=value.device, dtype=value.dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.conv1 is None:
            self._build(value.shape[1], value)
        assert self.bn1 is not None and self.conv_h is not None and self.conv_w is not None
        height, width = value.shape[2:]
        pooled_height = value.mean(dim=3, keepdim=True)
        pooled_width = value.mean(dim=2, keepdim=True).transpose(2, 3)
        encoded = self.activation(self.bn1(self.conv1(torch.cat([pooled_height, pooled_width], dim=2))))
        encoded_height, encoded_width = torch.split(encoded, [height, width], dim=2)
        attention_height = self.conv_h(encoded_height).sigmoid()
        attention_width = self.conv_w(encoded_width.transpose(2, 3)).sigmoid()
        return value * attention_height * attention_width


def register_ultralytics_modules() -> None:
    """Expose project-owned layers to Ultralytics' public YAML parser."""
    try:
        from ultralytics.nn import tasks
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required to build InsuLens custom models") from exc
    tasks.CoordAtt = CoordAtt


def create_yolo(model: str):
    """Create YOLO after registering the project-owned YAML modules."""
    register_ultralytics_modules()
    from ultralytics import YOLO

    return YOLO(model)
