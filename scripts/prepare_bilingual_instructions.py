"""Prepare offline Chinese translation placeholders for LIBERO instructions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.instruction_translator import translate_instructions
from src.data.libero_dataset import LiberoDataset
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint for translation preparation."""
    parser = argparse.ArgumentParser(description="Create English-to-Chinese instruction JSON.")
    parser.add_argument("--data-root", default="data/libero")
    parser.add_argument("--output", default="data/instructions_zh/libero_object_zh.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-chunk-size", type=int, default=8)
    args = parser.parse_args()
    setup_logging()

    dataset = LiberoDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        action_chunk_size=args.action_chunk_size,
        language_sampling="en",
    )
    instructions = [sample.instruction for sample in dataset.samples]
    mapping = translate_instructions(instructions, args.output)
    LOGGER.info("Prepared %d translation placeholders.", len(mapping))


if __name__ == "__main__":
    main()
