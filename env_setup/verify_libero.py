"""Verify LIBERO reset, step, and RGB rendering through the project wrapper."""

from __future__ import annotations

import sys

import numpy as np

from src.envs.libero_env import LiberoEnv
from src.utils.headless import enforce_headless


def main() -> int:
    """Run a short LIBERO smoke test."""
    status = enforce_headless(verify=False)
    print(f"Headless backend configured: {status.backend}")
    try:
        env = LiberoEnv(suite_name="libero_object", task_id=0, seed=0, record_video=True)
        obs = env.reset()
        action = np.zeros(7, dtype=np.float32)
        next_obs, reward, done, info = env.step(action)
        frame = next_obs["rgb"]
        env.close()
    except Exception as exc:
        print(f"ERROR: LIBERO smoke test failed: {type(exc).__name__}: {exc}")
        return 1
    print(
        "SUCCESS: LIBERO reset/step/render OK "
        f"obs_shape={obs['rgb'].shape} next_shape={frame.shape} reward={reward} done={done} info_keys={list(info.keys())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

