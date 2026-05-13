"""Verify MuJoCo headless rendering."""

from __future__ import annotations

import ctypes.util
import os
import subprocess
import sys
import textwrap


def main() -> int:
    """Render one offscreen MuJoCo frame with EGL, then OSMesa fallback."""
    print("Headless rendering diagnostics:")
    print(f"  libEGL: {ctypes.util.find_library('EGL')}")
    print(f"  libOpenGL: {ctypes.util.find_library('OpenGL')}")
    print(f"  libGL: {ctypes.util.find_library('GL')}")
    print(f"  NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES')}")
    print(f"  NVIDIA_DRIVER_CAPABILITIES: {os.environ.get('NVIDIA_DRIVER_CAPABILITIES')}")
    for backend in ("egl", "osmesa"):
        ok, message = _try_render(backend)
        if ok:
            print(message)
            return 0
        print(message)
    return 1


def _try_render(backend: str) -> tuple[bool, str]:
    """Try rendering with one MuJoCo backend."""
    env = os.environ.copy()
    env["MUJOCO_GL"] = backend
    env["PYOPENGL_PLATFORM"] = backend
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    env.setdefault("EGL_DEVICE_ID", "0")
    code = textwrap.dedent(
        """
        import mujoco

        xml = '''
        <mujoco>
          <worldbody>
            <light pos="0 0 2"/>
            <geom name="floor" type="plane" size="1 1 0.02" rgba="0.2 0.2 0.2 1"/>
            <geom name="ball" type="sphere" pos="0 0 0.2" size="0.1" rgba="0.8 0.1 0.1 1"/>
            <camera name="cam" pos="0 -1 0.6" xyaxes="1 0 0 0 0.6 1"/>
          </worldbody>
        </mujoco>
        '''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=64, width=64)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="cam")
        frame = renderer.render()
        renderer.close()
        print(f"SUCCESS: rendered frame shape={frame.shape}, dtype={frame.dtype}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return True, f"{result.stdout.strip()}, MUJOCO_GL={backend}"
    detail = (result.stderr or result.stdout).strip()
    return False, f"ERROR: MuJoCo {backend} rendering failed: {detail}"


if __name__ == "__main__":
    sys.exit(main())
