"""Shape-level regression test for project-owned Coordinate Attention."""

import torch

from insulens_perception.modeling import CoordAtt


def test_coordinate_attention_preserves_shape_and_builds_parameters():
    module = CoordAtt()
    output = module(torch.randn(2, 16, 21, 13))
    assert output.shape == (2, 16, 21, 13)
    assert sum(parameter.numel() for parameter in module.parameters()) > 0
