"""End-to-end Chinese voice instruction to simulated robot execution."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.eval.eval_libero import load_policy_from_checkpoint
from src.inference.asr_pipeline import WhisperASR
from src.inference.policy_runner import run_policy_rollout
from src.utils.headless import enforce_headless
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint for the voice demo."""
    parser = argparse.ArgumentParser(description="Run Chinese voice -> VLA -> LIBERO demo.")
    parser.add_argument("--audio", type=str, default=None, help="Input .wav file.")
    parser.add_argument("--mic", action="store_true", help="Record from microphone before ASR.")
    parser.add_argument("--model-config", default="configs/model/florence2_base.yaml")
    parser.add_argument("--checkpoint-path", default="checkpoints/bc_v100s/best")
    parser.add_argument("--suite-name", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--output-video", default="data/eval_videos/voice_demo.mp4")
    parser.add_argument("--output-json", default="data/eval_videos/voice_demo.json")
    args = parser.parse_args()
    setup_logging()
    enforce_headless()

    audio_path = args.audio
    if args.mic:
        raise NotImplementedError("Microphone recording is optional and not enabled in this scaffold.")
    if audio_path is None:
        raise ValueError("Provide --audio path, or implement optional --mic recording.")

    model_values = OmegaConf.to_container(OmegaConf.load(args.model_config), resolve=True)
    if not isinstance(model_values, dict):
        raise ValueError("Model config must contain a mapping.")
    asr = WhisperASR(model_name=args.whisper_model)
    asr_result = asr.transcribe(audio_path, language="zh")
    policy = load_policy_from_checkpoint(model_values, args.checkpoint_path)
    rollout = run_policy_rollout(
        policy=policy,
        instruction=asr_result.text,
        suite_name=args.suite_name,
        task_id=args.task_id,
        seed=args.seed,
        video_path=args.output_video,
        log_path=None,
    )
    payload = {
        "audio_path": str(audio_path),
        "asr": asdict(asr_result),
        "rollout": asdict(rollout),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Voice demo complete. Wrote %s and %s", args.output_video, output_json)


if __name__ == "__main__":
    main()
