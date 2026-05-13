from __future__ import annotations

from pathlib import Path

from src.data.instruction_translator import PLACEHOLDER_PREFIX, translate_instructions


def test_placeholder_translation_writes_mapping(tmp_path: Path) -> None:
    output = tmp_path / "translations.json"
    mapping = translate_instructions(["open the drawer", "open the drawer"], output)
    assert output.exists()
    assert mapping["open the drawer"].startswith(PLACEHOLDER_PREFIX)
    assert len(mapping) == 1

