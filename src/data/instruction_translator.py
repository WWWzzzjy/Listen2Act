"""Offline English-to-Chinese instruction translation helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

LOGGER = logging.getLogger(__name__)
PLACEHOLDER_PREFIX = "【待人工审核】"


class TranslationProvider(Protocol):
    """Protocol for optional external translation providers."""

    def translate(self, instruction: str) -> str:
        """Translate one instruction."""


class PlaceholderTranslator:
    """Deterministic placeholder translator for offline repository generation."""

    def translate(self, instruction: str) -> str:
        """Return a clearly marked placeholder translation.

        Args:
            instruction: English instruction.

        Returns:
            Placeholder Chinese text that should be manually reviewed.
        """
        return f"{PLACEHOLDER_PREFIX}{instruction}"


def translate_instructions(
    instructions: list[str],
    output_path: str | Path,
    provider: TranslationProvider | None = None,
) -> dict[str, str]:
    """Translate instructions and save an offline JSON mapping.

    Args:
        instructions: Unique English instructions.
        output_path: JSON output path.
        provider: Optional translation provider. A deterministic placeholder is used by default.

    Returns:
        English-to-Chinese mapping.
    """
    translator = provider or PlaceholderTranslator()
    mapping = {instruction: translator.translate(instruction) for instruction in sorted(set(instructions))}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote %d translated instructions to %s", len(mapping), path)
    return mapping


def load_instruction_mapping(path: str | Path) -> dict[str, str]:
    """Load an English-to-Chinese instruction mapping.

    Args:
        path: JSON mapping path.

    Returns:
        Translation mapping.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected translation mapping at {path}")
    return {str(key): str(value) for key, value in data.items()}

