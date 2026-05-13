"""Compatibility shim for Florence-2 on GPUs without FlashAttention-2.

This project targets NVIDIA V100/Volta, where FlashAttention-2 is unsupported.
Microsoft Florence-2's HuggingFace remote modeling file still contains optional
``flash_attn`` imports, and Transformers checks those imports before it can pick
the SDPA attention path. This shim satisfies that import check while making any
accidental FlashAttention call fail clearly.
"""

from __future__ import annotations

__version__ = "0.0.0"


def flash_attn_func(*_args: object, **_kwargs: object) -> None:
    """Raise because FlashAttention execution is disabled for V100."""
    raise RuntimeError(
        "flash_attn_func was called, but SimVoiceVLA disables FlashAttention-2. "
        "Use the configured PyTorch SDPA/eager attention path instead."
    )


def flash_attn_varlen_func(*_args: object, **_kwargs: object) -> None:
    """Raise because FlashAttention execution is disabled for V100."""
    raise RuntimeError(
        "flash_attn_varlen_func was called, but SimVoiceVLA disables FlashAttention-2. "
        "Use the configured PyTorch SDPA/eager attention path instead."
    )

