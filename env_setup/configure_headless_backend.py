"""Detect a working MuJoCo headless backend and write shell env exports."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

BACKENDS = ("egl", "osmesa")


def main() -> int:
    """Detect and persist the first working headless backend."""
    parser = argparse.ArgumentParser(description="Detect MuJoCo EGL/OSMesa backend.")
    parser.add_argument("--env-file", default=".env.headless", help="Shell env file to write.")
    args = parser.parse_args()

    for backend in BACKENDS:
        ok, message = try_backend(backend)
        print(message)
        if ok:
            write_env_file(Path(args.env_file), backend)
            print(f"Selected MuJoCo backend: {backend}")
            print(f"Wrote {args.env_file}")
            return 0
    print("ERROR: No working MuJoCo headless backend found.")
    return 1


def try_backend(backend: str) -> tuple[bool, str]:
    """Try rendering a tiny MuJoCo scene in a clean subprocess.

    Args:
        backend: ``egl`` or ``osmesa``.

    Returns:
        Success flag and diagnostic message.
    """
    code = textwrap.dedent(
        f"""
        import os
        os.environ["MUJOCO_GL"] = "{backend}"
        os.environ["PYOPENGL_PLATFORM"] = "{backend}"
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
        os.environ.setdefault("EGL_DEVICE_ID", "0")

        import mujoco

        xml = '''
        <mujoco>
          <worldbody>
            <light pos="0 0 2"/>
            <geom type="plane" size="1 1 0.02" rgba="0.2 0.2 0.2 1"/>
            <geom type="sphere" pos="0 0 0.2" size="0.1" rgba="0.8 0.1 0.1 1"/>
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
        print(f"shape={{frame.shape}}, dtype={{frame.dtype}}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return True, f"SUCCESS: MuJoCo {backend} render OK, {result.stdout.strip()}"
    detail = (result.stderr or result.stdout).strip().splitlines()[-1:]
    suffix = detail[0] if detail else "unknown error"
    return False, f"ERROR: MuJoCo {backend} render failed: {suffix}"


def write_env_file(path: Path, backend: str) -> None:
    """Write shell exports for the selected backend.

    Args:
        path: Output shell env file.
        backend: Selected MuJoCo backend.
    """
    path.write_text(
        "\n".join(
            [
                f"export MUJOCO_GL={backend}",
                f"export PYOPENGL_PLATFORM={backend}",
                'export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"',
                'export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())

