# Headless Rendering

MuJoCo can render through EGL, OSMesa, or GLFW. This project targets EGL because it supports GPU offscreen rendering on a remote server without an X server.

Required environment variables:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0
```

`src/utils/headless.py` sets these before simulator imports. Scripts that touch simulation call `enforce_headless()` before importing or constructing LIBERO environments.

Smoke test:

```bash
python env_setup/verify_headless_rendering.py
```

Common failures:

- `Could not initialize EGL`: host driver or EGL library is unavailable.
- Missing `libEGL.so`: install `libegl1` or the matching NVIDIA driver package.
- GLFW/X11 errors: a simulator import happened before `MUJOCO_GL=egl`; restart the process.

The codebase never calls `env.render(mode="human")`. Videos are saved from RGB arrays with ImageIO.

