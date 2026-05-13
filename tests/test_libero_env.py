from __future__ import annotations

import numpy as np

from src.envs.libero_env import LiberoEnv


class MockEnv:
    def __init__(self) -> None:
        self.closed = False

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def reset(self):
        return {"agentview_rgb": np.zeros((16, 16, 3), dtype=np.uint8)}

    def step(self, action):
        assert np.asarray(action).shape == (7,)
        obs = {"agentview_rgb": np.ones((16, 16, 3), dtype=np.uint8)}
        return obs, 1.0, True, {"success": True}

    def close(self) -> None:
        self.closed = True


def test_libero_env_wraps_mock_env() -> None:
    env = LiberoEnv(env_factory=MockEnv, record_video=True)
    obs = env.reset()
    assert obs["rgb"].shape == (16, 16, 3)
    next_obs, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
    assert next_obs["rgb"].shape == (16, 16, 3)
    assert reward == 1.0
    assert done is True
    assert info["success"] is True
    assert len(env.frames) == 2
    env.close()

