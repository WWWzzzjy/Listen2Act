"""Verify MuJoCo headless EGL rendering."""

from __future__ import annotations

import os
import sys


def main() -> int:
    """Render one offscreen MuJoCo frame."""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("EGL_DEVICE_ID", "0")
    try:
        import mujoco

        xml = """
        <mujoco>
          <worldbody>
            <light pos="0 0 2"/>
            <geom name="floor" type="plane" size="1 1 0.02" rgba="0.2 0.2 0.2 1"/>
            <geom name="ball" type="sphere" pos="0 0 0.2" size="0.1" rgba="0.8 0.1 0.1 1"/>
            <camera name="cam" pos="0 -1 0.6" xyaxes="1 0 0 0 0.6 1"/>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=64, width=64)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="cam")
        frame = renderer.render()
        renderer.close()
    except Exception as exc:
        print(f"ERROR: MuJoCo EGL rendering failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"SUCCESS: rendered frame shape={frame.shape}, dtype={frame.dtype}, MUJOCO_GL={os.environ['MUJOCO_GL']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

