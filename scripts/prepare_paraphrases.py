"""Prepare offline paraphrase placeholders for LIBERO instructions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.instruction_paraphraser import generate_paraphrases
from src.data.libero_dataset import LiberoDataset
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint for paraphrase preparation."""
    parser = argparse.ArgumentParser(description="Create paraphrase JSON placeholders.")
    parser.add_argument("--data-root", default="data/libero")
    parser.add_argument("--output", default="data/paraphrases/libero_object_paraphrases.json")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    setup_logging()

    dataset = LiberoDataset(data_root=args.data_root, language_sampling="en")
    instructions = [sample.instruction for sample in dataset.samples]
    mapping = generate_paraphrases(instructions, args.output, count=args.count)
    LOGGER.info("Prepared paraphrases for %d instructions.", len(mapping))


if __name__ == "__main__":
    main()
