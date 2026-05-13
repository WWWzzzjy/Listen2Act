#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi

PYTHON_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "${PYTHON_VERSION}" in
  3.10|3.11)
    ;;
  *)
    cat <<MSG
ERROR: SimVoiceVLA requires Python >=3.10,<3.12, but ${PYTHON_BIN} is Python ${PYTHON_VERSION}.

Install Python 3.10/3.11 or use Conda, then rerun one of:
  PYTHON_BIN=python3.10 bash env_setup/setup_remote_server.sh
  PYTHON_BIN=python3.11 bash env_setup/setup_remote_server.sh

Conda alternative:
  conda env create -f env_setup/conda_env.yaml
  conda activate simvoicevla
MSG
    exit 1
    ;;
esac

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==24.0 setuptools==70.2.0 wheel==0.43.0
python -m pip install --no-build-isolation -r requirements.txt

mkdir -p external
if [ ! -d external/LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
fi
python -m pip install --no-build-isolation -e external/LIBERO

cat > .env.headless <<'EOF'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0
EOF

echo "Setup complete. Run: source .venv/bin/activate && source .env.headless"
