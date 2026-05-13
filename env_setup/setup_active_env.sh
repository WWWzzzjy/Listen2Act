#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_VERSION="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "${PYTHON_VERSION}" in
  3.10|3.11)
    ;;
  *)
    cat <<MSG
ERROR: SimVoiceVLA requires Python >=3.10,<3.12, but active python is ${PYTHON_VERSION}.
Activate a Python 3.10/3.11 conda env first, then rerun:
  conda activate py310
  bash env_setup/setup_active_env.sh
MSG
    exit 1
    ;;
esac

python -m pip install --force-reinstall "pip==24.0" "setuptools==70.2.0" "wheel==0.43.0"
python -m pip install --no-build-isolation -c env_setup/libero_constraints.txt -r requirements.txt

mkdir -p external
if [ ! -d external/LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
fi
python -m pip install --no-build-isolation -c env_setup/libero_constraints.txt -r env_setup/libero_runtime_requirements.txt
python -m pip install --no-build-isolation --no-deps --force-reinstall "robosuite==1.4.1" "bddl==1.0.1"
python -m pip install --no-build-isolation -e external/LIBERO

python env_setup/init_libero_config.py --libero-root external/LIBERO --datasets data/libero
python env_setup/configure_headless_backend.py --env-file .env.headless

echo "Setup complete for active Python environment."
echo "Run: source .env.headless && python env_setup/verify_libero.py"
