#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==24.0 setuptools==70.2.0 wheel==0.43.0
python -m pip install -r requirements.txt

mkdir -p external
if [ ! -d external/LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
fi
python -m pip install -e external/LIBERO

cat > .env.headless <<'EOF'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0
EOF

echo "Setup complete. Run: source .venv/bin/activate && source .env.headless"

