"""Single-GPU distributed compatibility stubs."""

from __future__ import annotations

import torch


def is_main_process() -> bool:
    """Return whether the current process should write logs and checkpoints."""
    return True


def get_world_size() -> int:
    """Return the current distributed world size."""
    return 1


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module for single-GPU training.

    Args:
        model: Any PyTorch module.

    Returns:
        The same model.
    """
    return model

