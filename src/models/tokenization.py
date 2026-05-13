"""Tokenizer helpers for bilingual English and Chinese instructions."""

from __future__ import annotations

import logging
import re
from typing import Any

LOGGER = logging.getLogger(__name__)
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def contains_cjk(text: str) -> bool:
    """Return whether text contains Chinese/Japanese/Korean characters.

    Args:
        text: Input text.

    Returns:
        True when CJK characters are present.
    """
    return bool(CJK_PATTERN.search(text))


def ensure_tokenizer_padding(tokenizer: Any) -> None:
    """Ensure a HuggingFace tokenizer has a usable pad token.

    Args:
        tokenizer: HuggingFace tokenizer-like object.
    """
    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
            LOGGER.info("Set tokenizer pad_token to eos_token.")


def normalize_instruction(text: str) -> str:
    """Normalize whitespace while preserving Chinese characters.

    Args:
        text: Raw instruction.

    Returns:
        Normalized instruction string.
    """
    return " ".join(text.strip().split())

