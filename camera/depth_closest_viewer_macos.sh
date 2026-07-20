#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
SDK_DIR="${ORBBEC_SDK_DIR:-${WORKSPACE_DIR}/pyorbbecsdk-v1}"
SDK_PYTHON="${ORBBEC_SDK_PYTHON:-/opt/anaconda3/envs/gripperenv/bin/python}"

sudo env PYTHONPATH="${SDK_DIR}/install/lib" \
  "${SDK_PYTHON}" \
  "${SCRIPT_DIR}/depth_closest_viewer.py"
