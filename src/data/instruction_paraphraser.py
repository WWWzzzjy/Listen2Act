"""Offline paraphrase generation helpers for instruction robustness experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

LOGGER = logging.getLogger(__name__)
DEFAULT_PARAPHRASE_COUNT = 5


class ParaphraseProvider(Protocol):
    """Protocol for optional paraphrase providers."""

    def paraphrase(self, instruction: str, language: str, count: int) -> list[str]:
        """Generate paraphrases for one instruction."""


class PlaceholderParaphraser:
    """Deterministic placeholder paraphraser for initial repository scaffolding."""

    def paraphrase(self, instruction: str, language: str, count: int) -> list[str]:
        """Generate clearly marked placeholder variants.

        Args:
            instruction: Source instruction.
            language: Language code.
            count: Number of variants.

        Returns:
            Placeholder variants that should be replaced by reviewed text.
        """
        return [f"[{language} paraphrase {idx + 1}] {instruction}" for idx in range(count)]


def generate_paraphrases(
    instructions: list[str],
    output_path: str | Path,
    provider: ParaphraseProvider | None = None,
    count: int = DEFAULT_PARAPHRASE_COUNT,
) -> dict[str, dict[str, list[str]]]:
    """Generate and save English and Chinese paraphrase placeholders.

    Args:
        instructions: Unique base instructions.
        output_path: JSON output path.
        provider: Optional paraphrase provider.
        count: Number of variants per language.

    Returns:
        Mapping from source instruction to language-specific paraphrases.
    """
    paraphraser = provider or PlaceholderParaphraser()
    mapping = {
        instruction: {
            "en": paraphraser.paraphrase(instruction, "en", count),
            "zh": paraphraser.paraphrase(instruction, "zh", count),
        }
        for instruction in sorted(set(instructions))
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote paraphrases for %d instructions to %s", len(mapping), path)
    return mapping


def load_paraphrases(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """Load a paraphrase mapping.

    Args:
        path: JSON mapping path.

    Returns:
        Parsed paraphrase mapping.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected paraphrase mapping at {path}")
    return data

