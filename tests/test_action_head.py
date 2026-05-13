from __future__ import annotations

import torch

from src.models.action_head import ActionHead


def test_action_head_shape_and_ranges() -> None:
    head = ActionHead(input_dim=16, action_chunk_size=8)
    output = head(torch.randn(4, 16))
    assert output.shape == (4, 8, 7)
    assert torch.all(output[..., :6] <= 1.0)
    assert torch.all(output[..., :6] >= -1.0)
    assert torch.all(output[..., 6] >= 0.0)
    assert torch.all(output[..., 6] <= 1.0)

