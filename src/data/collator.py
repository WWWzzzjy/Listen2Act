"""Batch collation for VLA behavioral cloning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def collate_vla_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate LIBERO dataset samples into a training batch.

    Args:
        samples: Sequence of samples containing ``rgb``, ``action_chunk``,
            ``instruction``, and optionally ``state``.

    Returns:
        Dictionary with batched tensors and a list of instruction strings.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    images = torch.stack([_as_tensor(sample["rgb"]).float() for sample in samples], dim=0)
    actions = torch.stack([_as_tensor(sample["action_chunk"]).float() for sample in samples], dim=0)
    states = torch.stack([_as_tensor(sample.get("state", torch.empty(0))).float() for sample in samples], dim=0)
    instructions = [str(sample["instruction"]) for sample in samples]
    return {
        "images": images,
        "action_chunks": actions,
        "states": states,
        "instructions": instructions,
    }


def _as_tensor(value: Any) -> torch.Tensor:
    """Convert a sample value to a tensor."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)

