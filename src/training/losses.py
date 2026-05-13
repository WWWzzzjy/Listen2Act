"""Loss functions for action chunk prediction."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def action_mse_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MSE loss on full action chunks.

    Args:
        predicted_actions: Predicted tensor with shape ``(B, H, 7)``.
        target_actions: Target tensor with shape ``(B, H, 7)``.
        mask: Optional broadcastable mask.

    Returns:
        Scalar loss. A non-finite tensor is returned and logged if inputs are invalid,
        allowing the training loop to skip the batch gracefully.
    """
    try:
        if predicted_actions.shape != target_actions.shape:
            raise ValueError(
                f"Action shape mismatch: predicted={predicted_actions.shape}, "
                f"target={target_actions.shape}"
            )
        loss = F.mse_loss(predicted_actions, target_actions, reduction="none")
        if mask is not None:
            loss = loss * mask.to(loss.device, dtype=loss.dtype)
            denom = mask.to(loss.device, dtype=loss.dtype).sum().clamp_min(1.0)
            return loss.sum() / denom
        return loss.mean()
    except Exception as exc:
        LOGGER.exception("Action MSE loss failed: %s", exc)
        return predicted_actions.sum() * 0.0 + torch.tensor(
            float("nan"), device=predicted_actions.device, dtype=predicted_actions.dtype
        )

