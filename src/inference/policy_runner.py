"""Policy rollout helpers for LIBERO simulation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.envs.libero_env import LiberoEnv
from src.envs.render_recorder import RenderRecorder

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_STEPS = 600


@dataclass
class RolloutLog:
    """JSON-serializable rollout log."""

    instruction: str
    actions: list[list[float]]
    rewards: list[float]
    success: bool
    steps: int
    video_path: str | None
    error: str | None = None


def run_policy_rollout(
    policy: Any,
    instruction: str,
    suite_name: str = "libero_object",
    task_id: int = 0,
    seed: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
    video_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> RolloutLog:
    """Run one policy rollout and optionally save MP4 plus JSON logs.

    Args:
        policy: Object exposing ``predict``.
        instruction: Natural-language instruction.
        suite_name: LIBERO suite name.
        task_id: Task ID.
        seed: Environment seed.
        max_steps: Maximum rollout steps.
        video_path: Optional MP4 output path.
        log_path: Optional JSON output path.

    Returns:
        Rollout log.
    """
    env = LiberoEnv(suite_name=suite_name, task_id=task_id, seed=seed, record_video=video_path is not None)
    actions: list[list[float]] = []
    rewards: list[float] = []
    success = False
    error: str | None = None
    if hasattr(policy, "reset_rollout"):
        policy.reset_rollout()
    try:
        obs = env.reset(seed=seed)
        for step in range(max_steps):
            action = policy.predict(obs["rgb"], instruction)
            action_array = _to_numpy_action(action)
            actions.append(action_array.astype(float).tolist())
            obs, reward, done, info = env.step(action_array)
            rewards.append(float(reward))
            if done:
                success = bool(info.get("success", reward > 0.0))
                break
        steps = len(actions)
    except Exception as exc:
        LOGGER.exception("Rollout failed: %s", exc)
        error = str(exc)
        steps = len(actions)
    finally:
        saved_video = None
        if video_path is not None and env.frames:
            recorder = RenderRecorder()
            recorder.extend(env.frames)
            saved = recorder.save(video_path)
            saved_video = str(saved) if saved else None
        env.close()

    rollout = RolloutLog(
        instruction=instruction,
        actions=actions,
        rewards=rewards,
        success=success,
        steps=steps,
        video_path=saved_video,
        error=error,
    )
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(rollout), handle, ensure_ascii=False, indent=2)
    return rollout


def _to_numpy_action(action: Any) -> np.ndarray:
    """Convert a policy action to NumPy."""
    if isinstance(action, torch.Tensor):
        return action.detach().cpu().numpy().astype(np.float32)
    return np.asarray(action, dtype=np.float32)

