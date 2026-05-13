from __future__ import annotations

from src.utils.headless import enforce_headless


def test_enforce_headless_sets_egl_env(monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    status = enforce_headless(verify=False)
    assert status.backend == "egl"
    assert status.verified is False
    assert status.message

