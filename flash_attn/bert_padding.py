"""Compatibility padding helpers for optional Florence-2 FlashAttention imports."""

from __future__ import annotations

import torch


def index_first_axis(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Index the first axis, matching the flash-attn helper signature."""
    return input_tensor.index_select(0, indices)


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Pad an unpadded tensor back to ``(batch, seqlen, ...)``.

    This function exists only so Florence-2's optional imports resolve. The
    project config keeps FlashAttention disabled, so normal training should not
    call this path.
    """
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(0, indices, hidden_states)
    return output.view(batch, seqlen, *hidden_states.shape[1:])


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remove padded tokens, matching the flash-attn helper signature."""
    batch, seqlen = attention_mask.shape
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    unpadded = hidden_states.reshape(batch * seqlen, *hidden_states.shape[2:]).index_select(0, indices)
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    return unpadded, indices, cu_seqlens, max_seqlen

