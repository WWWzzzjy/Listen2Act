#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET="${LIBERO_DATASET:-libero_object}"
TARGET_DIR="${LIBERO_DATA_DIR:-${PROJECT_ROOT}/data/libero}"

mkdir -p "${TARGET_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/LIBERO:${PYTHONPATH:-}"

python - <<PY
from __future__ import annotations

from pathlib import Path

from env_setup.init_libero_config import write_libero_config

project_root = Path("${PROJECT_ROOT}").resolve()
download_dir = Path("${TARGET_DIR}").resolve()
dataset = "${DATASET}"

write_libero_config(
    libero_root=project_root / "external" / "LIBERO",
    datasets=download_dir,
    project_root=project_root,
)

from libero.libero.utils.download_utils import check_libero_dataset, libero_dataset_download

print(f"Downloading LIBERO dataset '{dataset}' to {download_dir}")
libero_dataset_download(
    datasets=dataset,
    download_dir=str(download_dir),
    check_overwrite=False,
    use_huggingface=True,
)
check_libero_dataset(download_dir=str(download_dir))
PY

echo "Downloaded files:"
find "${TARGET_DIR}" -name "*.hdf5" -o -name "*.h5" | head -20
