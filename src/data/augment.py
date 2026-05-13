"""Instruction and image augmentation utilities."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch

DEFAULT_EN_PROBABILITY = 0.5
DEFAULT_ZH_PROBABILITY = 0.5


def sample_instruction(
    english_instruction: str,
    translations: Mapping[str, str] | None = None,
    paraphrases: Mapping[str, Any] | None = None,
    language_sampling: str = "en",
    use_paraphrases: bool = False,
    rng: random.Random | None = None,
) -> str:
    """Sample an English or Chinese instruction variant.

    Args:
        english_instruction: Original LIBERO instruction.
        translations: Mapping from English instruction to Chinese translation.
        paraphrases: Mapping from instruction or task key to paraphrase variants.
        language_sampling: One of ``en``, ``zh``, or ``bilingual``.
        use_paraphrases: Whether to sample paraphrases when available.
        rng: Optional random generator.

    Returns:
        Selected instruction string.
    """
    generator = rng or random
    language = _sample_language(language_sampling, generator)
    base = english_instruction
    if language == "zh" and translations:
        base = translations.get(english_instruction, english_instruction)

    if not use_paraphrases or not paraphrases:
        return base

    variants = _extract_variants(paraphrases.get(english_instruction), language)
    if variants:
        return generator.choice(variants)
    return base


def maybe_apply_image_noise(image: torch.Tensor, std: float = 0.0) -> torch.Tensor:
    """Apply optional Gaussian noise to an image tensor.

    Args:
        image: Tensor with shape ``(C, H, W)``.
        std: Noise standard deviation; no noise is applied when zero.

    Returns:
        Augmented image tensor.
    """
    if std <= 0.0:
        return image
    return image + torch.randn_like(image) * std


def _sample_language(language_sampling: str, rng: random.Random | random.Random) -> str:
    """Sample a language code from a small policy string."""
    normalized = language_sampling.lower()
    if normalized in {"en", "english"}:
        return "en"
    if normalized in {"zh", "chinese", "cn"}:
        return "zh"
    if normalized in {"bilingual", "mixed"}:
        return "zh" if rng.random() < DEFAULT_ZH_PROBABILITY else "en"
    return "en"


def _extract_variants(entry: Any, language: str) -> list[str]:
    """Extract paraphrase variants from supported JSON layouts."""
    if entry is None:
        return []
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, Sequence) and not isinstance(entry, bytes):
        return [str(item) for item in entry]
    if isinstance(entry, Mapping):
        variants = entry.get(language) or entry.get("instructions") or entry.get("variants")
        if isinstance(variants, str):
            return [variants]
        if isinstance(variants, Sequence) and not isinstance(variants, bytes):
            return [str(item) for item in variants]
    return []

