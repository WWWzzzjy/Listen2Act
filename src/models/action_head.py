"""Action prediction head for chunked 7-DoF robot actions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

DEFAULT_ACTION_CHUNK_SIZE = 8
DEFAULT_ACTION_DIM = 7
DEFAULT_HIDDEN_DIM = 1024
DEFAULT_DEPTH = 3
DEFAULT_CONTINUOUS_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


class ActionHead(nn.Module):
    """MLP action head that predicts fixed-length action chunks."""

    def __init__(
        self,
        input_dim: int,
        action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
        action_dim: int = DEFAULT_ACTION_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        depth: int = DEFAULT_DEPTH,
        continuous_scales: Sequence[float] = DEFAULT_CONTINUOUS_SCALE,
    ) -> None:
        """Initialize the action head.

        Args:
            input_dim: Pooled VLM feature dimension.
            action_chunk_size: Number of actions predicted per forward pass.
            action_dim: Action dimension; SimVoiceVLA uses 7.
            hidden_dim: Hidden layer width.
            depth: Number of linear layers including the output layer.
            continuous_scales: Per-dimension scales for xyz and rpy dimensions.
        """
        super().__init__()
        if action_dim != DEFAULT_ACTION_DIM:
            raise ValueError("ActionHead currently expects 7-DoF actions.")
        if depth < 2:
            raise ValueError("ActionHead depth must be at least 2.")
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, action_chunk_size * action_dim))
        self.net = nn.Sequential(*layers)
        scales = torch.tensor(list(continuous_scales), dtype=torch.float32)
        if scales.numel() != DEFAULT_ACTION_DIM - 1:
            raise ValueError("continuous_scales must contain six values for xyz+rpy.")
        self.register_buffer("continuous_scales", scales.view(1, 1, DEFAULT_ACTION_DIM - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict scaled action chunks.

        Args:
            features: Tensor of shape ``(batch, input_dim)``.

        Returns:
            Tensor of shape ``(batch, H, 7)``. Continuous dimensions are tanh-scaled
            and the gripper dimension is sigmoid-scaled.
        """
        raw = self.net(features)
        raw = raw.view(features.shape[0], self.action_chunk_size, self.action_dim)
        continuous = torch.tanh(raw[..., : DEFAULT_ACTION_DIM - 1]) * self.continuous_scales
        gripper = torch.sigmoid(raw[..., DEFAULT_ACTION_DIM - 1 : DEFAULT_ACTION_DIM])
        return torch.cat([continuous, gripper], dim=-1)

