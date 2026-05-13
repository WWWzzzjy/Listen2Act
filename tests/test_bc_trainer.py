from __future__ import annotations

import torch
from torch import nn

from src.training.bc_trainer import _cast_trainable_parameters_to_fp32


def test_trainer_casts_trainable_parameters_to_fp32() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    model[0].requires_grad_(False)
    model.half()

    _cast_trainable_parameters_to_fp32(model)

    assert model[0].weight.dtype == torch.float16
    assert model[1].weight.dtype == torch.float32
