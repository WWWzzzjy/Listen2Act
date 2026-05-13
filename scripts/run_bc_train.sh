#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -m src.training.bc_trainer \
  --training-config configs/training/bc_v100s.yaml \
  --model-config configs/model/florence2_base.yaml \
  --data-config configs/data/libero_object.yaml

