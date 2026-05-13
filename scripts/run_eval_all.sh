#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -m src.eval.eval_libero --eval-config configs/eval/libero_eval.yaml --model-config configs/model/florence2_base.yaml
python -m src.eval.eval_bilingual --eval-config configs/eval/libero_eval.yaml --model-config configs/model/florence2_base.yaml
python -m src.eval.eval_paraphrase --eval-config configs/eval/libero_eval.yaml --model-config configs/model/florence2_base.yaml
