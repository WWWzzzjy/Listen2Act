#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${PROJECT_ROOT}/data/libero"
mkdir -p "${TARGET_DIR}"

if [ -z "${LIBERO_DEMO_URL:-}" ]; then
  cat <<'MSG'
LIBERO_DEMO_URL is not set.

Download the official LIBERO demonstration archive from the LIBERO project page,
then either extract it into data/libero manually or rerun this script with:

  LIBERO_DEMO_URL=https://.../libero_object_demos.tar.gz scripts/download_libero_demos.sh
MSG
  exit 0
fi

ARCHIVE="${TARGET_DIR}/libero_demos.tar.gz"
curl -L "${LIBERO_DEMO_URL}" -o "${ARCHIVE}"
tar -xzf "${ARCHIVE}" -C "${TARGET_DIR}"
echo "LIBERO demos extracted to ${TARGET_DIR}"

