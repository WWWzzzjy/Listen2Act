"""Evaluate a trained policy on LIBERO tasks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.envs.libero_env import LiberoEnv
from src.envs.render_recorder import RenderRecorder
from src.models.florence2_vla import Florence2VLA
from src.utils.checkpointing import load_action_head_state
from src.utils.headless import enforce_headless
from src.utils.logging_utils import setup_logging
from src.utils.seeding import seed_everything

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_EPISODE_STEPS = 600
DEFAULT_EPISODES_PER_TASK = 20


@dataclass
class EpisodeResult:
    """Result metadata for one evaluation episode."""

    success: bool
    reward: float
    steps: int
    video_path: str | None
    error: str | None = None


def run_policy_episode(
    policy: Any,
    env: LiberoEnv,
    instruction: str,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
) -> EpisodeResult:
    """Run one policy rollout in a LIBERO environment.

    Args:
        policy: Object exposing ``predict(image, instruction)``.
        env: LIBERO environment wrapper.
        instruction: Language instruction.
        max_episode_steps: Safety step limit.

    Returns:
        Episode result.
    """
    total_reward = 0.0
    if hasattr(policy, "reset_rollout"):
        policy.reset_rollout()
    try:
        obs = env.reset()
        info: dict[str, Any] = {}
        for step in range(max_episode_steps):
            action = policy.predict(obs["rgb"], instruction)
            action_array = _to_numpy_action(action)
            obs, reward, done, info = env.step(action_array)
            total_reward += reward
            if done:
                success = bool(info.get("success", reward > 0.0))
                return EpisodeResult(success=success, reward=total_reward, steps=step + 1, video_path=None)
        return EpisodeResult(
            success=bool(info.get("success", False)),
            reward=total_reward,
            steps=max_episode_steps,
            video_path=None,
        )
    except Exception as exc:
        LOGGER.exception("Policy episode failed: %s", exc)
        return EpisodeResult(success=False, reward=total_reward, steps=0, video_path=None, error=str(exc))


def evaluate_task(
    policy: Any,
    suite_name: str,
    task_id: int,
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    output_dir: str | Path = "data/eval_videos",
    save_videos: bool = True,
    fps: int = 20,
    video_quality: int = 5,
    seed: int = 0,
    instruction_override: str | None = None,
) -> dict[str, Any]:
    """Evaluate one LIBERO task.

    Args:
        policy: Policy object.
        suite_name: LIBERO suite name.
        task_id: Task id.
        episodes_per_task: Number of episodes.
        max_episode_steps: Maximum rollout length.
        output_dir: Video output directory.
        save_videos: Whether to save MP4 files.
        fps: Video FPS.
        video_quality: H.264 quality value.
        seed: Base seed.
        instruction_override: Optional instruction instead of official task text.

    Returns:
        JSON-serializable task result.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[EpisodeResult] = []
    for episode_idx in range(episodes_per_task):
        env: LiberoEnv | None = None
        try:
            env = LiberoEnv(
                suite_name=suite_name,
                task_id=task_id,
                seed=seed + episode_idx,
                record_video=save_videos,
            )
            instruction = instruction_override or env.instruction
            result = run_policy_episode(policy, env, instruction, max_episode_steps)
            if save_videos and env.frames:
                recorder = RenderRecorder(fps=fps, quality=video_quality)
                recorder.extend(env.frames)
                video_file = output_path / f"task_{task_id:03d}_episode_{episode_idx:03d}.mp4"
                saved = recorder.save(video_file)
                result.video_path = str(saved) if saved else None
            results.append(result)
        except Exception as exc:
            LOGGER.exception("Evaluation task=%d episode=%d failed: %s", task_id, episode_idx, exc)
            results.append(EpisodeResult(False, 0.0, 0, None, str(exc)))
        finally:
            if env is not None:
                env.close()

    successes = sum(1 for result in results if result.success)
    return {
        "task_id": task_id,
        "success_rate": successes / max(1, len(results)),
        "n_episodes": len(results),
        "episodes": [result.__dict__ for result in results],
    }


def load_policy_from_checkpoint(
    model_config: dict[str, Any],
    checkpoint_path: str | Path | None,
    device: str | torch.device | None = None,
) -> Florence2VLA:
    """Load a Florence2VLA policy and optional checkpoint state.

    Args:
        model_config: Model YAML mapping.
        checkpoint_path: Optional checkpoint directory.
        device: Optional device.

    Returns:
        Loaded policy in eval mode.
    """
    policy = Florence2VLA.from_config(model_config)
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            _try_load_lora_adapter(policy, checkpoint)
            load_action_head_state(policy, checkpoint, strict=False)
        else:
            LOGGER.warning("Checkpoint path does not exist: %s", checkpoint)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy.to(resolved_device)
    policy.eval()
    return policy


def _try_load_lora_adapter(policy: Florence2VLA, checkpoint: Path) -> None:
    """Best-effort LoRA adapter load for PEFT-backed Florence models."""
    backbone = getattr(policy, "backbone", None)
    if backbone is None:
        return
    try:
        if hasattr(backbone, "load_adapter"):
            backbone.load_adapter(str(checkpoint), adapter_name="default")
            LOGGER.info("Loaded LoRA adapter from %s", checkpoint)
    except Exception as exc:
        LOGGER.warning("LoRA adapter load failed, keeping initialized adapters: %s", exc)


def _to_numpy_action(action: Any) -> np.ndarray:
    """Convert a model action to a float32 NumPy vector."""
    if isinstance(action, torch.Tensor):
        return action.detach().cpu().numpy().astype(np.float32)
    return np.asarray(action, dtype=np.float32)


def main() -> None:
    """CLI entrypoint for LIBERO evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate SimVoiceVLA on LIBERO.")
    parser.add_argument("--eval-config", default="configs/eval/libero_eval.yaml")
    parser.add_argument("--model-config", default="configs/model/florence2_base.yaml")
    args = parser.parse_args()
    setup_logging()
    enforce_headless()

    eval_values = OmegaConf.to_container(OmegaConf.load(args.eval_config), resolve=True)
    model_values = OmegaConf.to_container(OmegaConf.load(args.model_config), resolve=True)
    if not isinstance(eval_values, dict) or not isinstance(model_values, dict):
        raise ValueError("Config files must contain mappings.")
    seed_everything(int(eval_values.get("seed", 123)))
    policy = load_policy_from_checkpoint(model_values, eval_values.get("checkpoint_path"))
    task_ids = eval_values.get("task_ids") or list(range(int(eval_values.get("num_tasks", 10))))
    all_results = {}
    for task_id in task_ids:
        task_result = evaluate_task(
            policy=policy,
            suite_name=str(eval_values.get("suite_name", "libero_object")),
            task_id=int(task_id),
            episodes_per_task=int(eval_values.get("episodes_per_task", DEFAULT_EPISODES_PER_TASK)),
            max_episode_steps=int(eval_values.get("max_episode_steps", DEFAULT_MAX_EPISODE_STEPS)),
            output_dir=eval_values.get("output_dir", "data/eval_videos"),
            save_videos=bool(eval_values.get("save_videos", True)),
            fps=int(eval_values.get("fps", 20)),
            video_quality=int(eval_values.get("video_quality", 5)),
            seed=int(eval_values.get("seed", 123)),
        )
        all_results[str(task_id)] = task_result
    results_path = Path(str(eval_values.get("results_path", "data/eval_videos/libero_eval_results.json")))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote evaluation results to %s", results_path)


if __name__ == "__main__":
    main()
