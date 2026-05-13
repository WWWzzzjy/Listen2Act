"""Checkpoint save and load helpers for LoRA VLA policies."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)
ACTION_HEAD_FILE = "action_head.pt"
TRAINER_STATE_FILE = "trainer_state.json"


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    step: int,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Save LoRA adapters when available plus the action head state.

    Args:
        model: Policy module.
        checkpoint_dir: Directory to create or update.
        step: Training step.
        metrics: Optional scalar metrics.

    Returns:
        The checkpoint directory path.
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(path)
        except Exception as exc:
            LOGGER.warning("Model save_pretrained failed, continuing with action head: %s", exc)

    action_head = getattr(model, "action_head", None)
    if action_head is not None:
        torch.save(action_head.state_dict(), path / ACTION_HEAD_FILE)
    else:
        torch.save(model.state_dict(), path / "model_state.pt")

    state = {"step": step, "metrics": metrics or {}}
    with (path / TRAINER_STATE_FILE).open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    return path


def load_action_head_state(model: torch.nn.Module, checkpoint_dir: str | Path, strict: bool = True) -> None:
    """Load an action head state dict if present.

    Args:
        model: Policy module with an ``action_head`` attribute.
        checkpoint_dir: Checkpoint directory.
        strict: Whether to enforce exact parameter matching.
    """
    path = Path(checkpoint_dir) / ACTION_HEAD_FILE
    if not path.exists():
        LOGGER.warning("Action head checkpoint not found at %s", path)
        return
    action_head = getattr(model, "action_head", None)
    if action_head is None:
        raise AttributeError("Model has no action_head attribute.")
    state = torch.load(path, map_location="cpu")
    action_head.load_state_dict(state, strict=strict)


def load_trainer_state(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load trainer state metadata.

    Args:
        checkpoint_dir: Checkpoint directory.

    Returns:
        Trainer state dictionary, or an empty dict if unavailable.
    """
    path = Path(checkpoint_dir) / TRAINER_STATE_FILE
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

