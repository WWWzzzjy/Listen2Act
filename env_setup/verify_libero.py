"""Verify LIBERO reset, step, and RGB rendering through the project wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LIBERO_ROOT = PROJECT_ROOT / "external" / "LIBERO"
if LIBERO_ROOT.exists() and str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

from env_setup.init_libero_config import write_libero_config
from src.envs.libero_env import LiberoEnv
from src.utils.headless import enforce_headless


def main() -> int:
    """Run a short LIBERO smoke test."""
    status = enforce_headless(verify=False)
    print(f"Headless backend configured: {status.backend}")
    config_path = write_libero_config(project_root=PROJECT_ROOT)
    print(f"LIBERO config initialized: {config_path}")
    ok, message = _verify_libero_imports()
    print(message)
    if not ok:
        return 1
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


def _verify_libero_imports() -> tuple[bool, str]:
    """Verify official LIBERO modules and report the exact import error."""
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        suites = list(benchmark.get_benchmark_dict().keys())[:10]
    except Exception as exc:
        return (
            False,
            "ERROR: LIBERO import failed before env creation. "
            f"{type(exc).__name__}: {exc}",
        )
    return True, f"LIBERO imports OK. suites={suites}, env={OffScreenRenderEnv.__name__}"


if __name__ == "__main__":
    sys.exit(main())
