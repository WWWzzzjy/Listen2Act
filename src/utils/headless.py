"""Headless rendering configuration for MuJoCo, robosuite, and LIBERO."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)
DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 64


@dataclass(frozen=True)
class HeadlessStatus:
    """Result of configuring and optionally probing a headless renderer."""

    backend: str
    verified: bool
    message: str


def enforce_headless(
    backend: str | None = None,
    fallback_backend: str = "osmesa",
    verify: bool = False,
) -> HeadlessStatus:
    """Set headless MuJoCo environment variables before simulator imports.

    Args:
        backend: Preferred MuJoCo OpenGL backend. When omitted, an existing
            ``MUJOCO_GL`` value is respected before falling back to EGL.
        fallback_backend: Backend used if verification of the preferred backend fails.
        verify: Whether to import MuJoCo and render a tiny offscreen context.

    Returns:
        A status object describing the selected backend.
    """
    selected = (backend or os.environ.get("MUJOCO_GL") or "egl").lower()
    _set_backend_env(selected)
    if not verify:
        return HeadlessStatus(backend=selected, verified=False, message="Headless env vars set.")

    ok, message = _probe_mujoco_context()
    if ok:
        return HeadlessStatus(backend=selected, verified=True, message=message)

    LOGGER.warning("Preferred MuJoCo backend %s failed: %s", selected, message)
    fallback = fallback_backend.lower()
    _set_backend_env(fallback)
    ok, fallback_message = _probe_mujoco_context()
    if ok:
        return HeadlessStatus(
            backend=fallback,
            verified=True,
            message=f"Fallback backend {fallback} verified after {selected} failed.",
        )
    return HeadlessStatus(
        backend=fallback,
        verified=False,
        message=f"{selected} failed: {message}; {fallback} failed: {fallback_message}",
    )


def _set_backend_env(backend: str) -> None:
    """Set environment variables consumed by MuJoCo and PyOpenGL."""
    os.environ["MUJOCO_GL"] = backend
    os.environ["PYOPENGL_PLATFORM"] = backend
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("EGL_DEVICE_ID", "0")


def _probe_mujoco_context() -> tuple[bool, str]:
    """Create and release a tiny MuJoCo GL context."""
    try:
        import mujoco  # type: ignore

        context = mujoco.GLContext(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        context.make_current()
        context.free()
    except Exception as exc:  # pragma: no cover - depends on host drivers.
        return False, f"{type(exc).__name__}: {exc}"
    return True, "MuJoCo offscreen context initialized successfully."
