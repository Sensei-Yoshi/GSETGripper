#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
SDK_DIR="${ORBBEC_SDK_DIR:-${WORKSPACE_DIR}/pyorbbecsdk-v1}"

sudo env PYTHONPATH="${SDK_DIR}/install/lib" \
  "${SDK_DIR}/venv/bin/python" \
  "${SDK_DIR}/examples/depth_viewer.py"
