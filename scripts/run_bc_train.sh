#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python env_setup/assert_python_version.py
python -m src.training.bc_trainer \
  --training-config "${TRAINING_CONFIG:-configs/training/bc_v100s.yaml}" \
  --model-config "${MODEL_CONFIG:-configs/model/florence2_base.yaml}" \
  --data-config "${DATA_CONFIG:-configs/data/libero_object.yaml}"
