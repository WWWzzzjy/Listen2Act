"""Logging helpers for console, JSONL, and optional Weights & Biases."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once.

    Args:
        level: Python logging level.
    """
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)


class JsonlLogger:
    """Append structured scalar logs to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the logger.

        Args:
            path: Output JSONL file path.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        """Append one JSON object with a timestamp.

        Args:
            payload: Serializable metrics dictionary.
        """
        row = {"time": time.time(), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class WandbLogger:
    """Thin optional wrapper around Weights & Biases."""

    def __init__(self, enabled: bool, project: str, config: dict[str, Any] | None = None) -> None:
        """Create an optional WandB run.

        Args:
            enabled: Whether to initialize WandB.
            project: WandB project name.
            config: Optional run configuration.
        """
        self.enabled = enabled
        self._run = None
        if enabled:
            try:
                import wandb

                self._run = wandb.init(project=project, config=config)
            except Exception as exc:  # pragma: no cover - external service.
                self.enabled = False
                LOGGER.warning("WandB initialization failed: %s", exc)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to WandB when enabled.

        Args:
            payload: Metrics dictionary.
            step: Optional global step.
        """
        if self.enabled and self._run is not None:
            self._run.log(payload, step=step)

    def finish(self) -> None:
        """Finish the WandB run when enabled."""
        if self.enabled and self._run is not None:
            self._run.finish()

