"""Whisper-based Chinese ASR wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_LANGUAGE = "zh"


@dataclass(frozen=True)
class ASRResult:
    """ASR transcription result."""

    text: str
    language: str
    segments: list[dict[str, Any]]


class WhisperASR:
    """Lazy-loading wrapper around the ``openai-whisper`` package."""

    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL, device: str | None = None) -> None:
        """Initialize the ASR wrapper.

        Args:
            model_name: Whisper model name.
            device: Optional torch device string.
        """
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    def transcribe(self, audio_path: str | Path, language: str = DEFAULT_LANGUAGE) -> ASRResult:
        """Transcribe a WAV file.

        Args:
            audio_path: Input audio path.
            language: Whisper language code.

        Returns:
            ASR result.
        """
        model = self._load_model()
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        result = model.transcribe(str(path), language=language, fp16=self.device != "cpu")
        text = str(result.get("text", "")).strip()
        segments = result.get("segments", [])
        LOGGER.info("ASR transcript: %s", text)
        return ASRResult(text=text, language=language, segments=segments)

    def _load_model(self) -> Any:
        """Load Whisper on first use."""
        if self._model is not None:
            return self._model
        try:
            import whisper
        except ImportError as exc:
            raise ImportError("Install openai-whisper to use the voice demo.") from exc
        self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

