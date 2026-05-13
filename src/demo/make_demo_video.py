"""Compose multiple rollout videos into a demo reel."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def compose_demo_video(video_paths: list[str | Path], log_paths: list[str | Path], output_path: str | Path) -> Path:
    """Compose a final demo reel with best-effort text overlays.

    Args:
        video_paths: Source MP4 paths.
        log_paths: JSON logs containing ASR and rollout metadata.
        output_path: Output MP4 path.

    Returns:
        Output path.
    """
    try:
        from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip, concatenate_videoclips
    except ImportError as exc:
        raise ImportError("Install moviepy to compose demo videos.") from exc

    clips = []
    logs = [_load_json(path) for path in log_paths]
    for idx, video_path in enumerate(video_paths):
        clip = VideoFileClip(str(video_path))
        overlay_text = _format_overlay(logs[idx] if idx < len(logs) else {})
        try:
            text_clip = (
                TextClip(overlay_text, fontsize=28, color="white", bg_color="black", method="caption")
                .set_duration(clip.duration)
                .set_position(("center", "bottom"))
            )
            clip = CompositeVideoClip([clip, text_clip])
        except Exception as exc:
            LOGGER.warning("Text overlay failed for %s: %s", video_path, exc)
        clips.append(clip)
    final_clip = concatenate_videoclips(clips, method="compose")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_clip.write_videofile(str(output), fps=20, codec="libx264", audio=False, logger=None)
    for clip in clips:
        clip.close()
    final_clip.close()
    LOGGER.info("Wrote demo reel to %s", output)
    return output


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON log file."""
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _format_overlay(log: dict[str, Any]) -> str:
    """Format a compact text overlay for one clip."""
    asr_text = str(log.get("asr", {}).get("text", ""))
    success = str(log.get("rollout", {}).get("success", "unknown"))
    return f"ASR: {asr_text}\nSuccess: {success}"


def main() -> None:
    """CLI entrypoint for demo reel composition."""
    parser = argparse.ArgumentParser(description="Compose SimVoiceVLA demo videos.")
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--output", default="data/eval_videos/demo_reel.mp4")
    args = parser.parse_args()
    setup_logging()
    compose_demo_video(args.videos, args.logs, args.output)


if __name__ == "__main__":
    main()
