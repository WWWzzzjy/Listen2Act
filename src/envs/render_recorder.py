"""Save headless RGB rollouts as MP4 videos."""

from __future__ import annotations

import logging
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_FPS = 20
DEFAULT_QUALITY = 5


class RenderRecorder:
    """Accumulate frames and write a compressed MP4 file."""

    def __init__(self, fps: int = DEFAULT_FPS, quality: int = DEFAULT_QUALITY) -> None:
        """Initialize the recorder.

        Args:
            fps: Video frames per second.
            quality: ImageIO H.264 quality value.
        """
        self.fps = fps
        self.quality = quality
        self.frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> None:
        """Append one RGB frame.

        Args:
            frame: HWC uint8 RGB frame.
        """
        self.frames.append(np.asarray(frame, dtype=np.uint8))

    def extend(self, frames: list[np.ndarray]) -> None:
        """Append multiple frames.

        Args:
            frames: RGB frame list.
        """
        for frame in frames:
            self.add_frame(frame)

    def save(self, path: str | Path) -> Path | None:
        """Write frames to an MP4 file.

        Args:
            path: Output video path.

        Returns:
            Output path, or None if saving failed.
        """
        if not self.frames:
            LOGGER.warning("No frames available for video output: %s", path)
            return None
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            imageio.mimsave(
                output,
                self.frames,
                fps=self.fps,
                quality=self.quality,
                codec="libx264",
                macro_block_size=1,
            )
        except Exception as exc:
            LOGGER.exception("Failed to save video %s: %s", output, exc)
            return None
        LOGGER.info("Saved video to %s", output)
        return output

