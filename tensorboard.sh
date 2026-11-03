#!/usr/bin/env bash
# Serve this project's TensorBoard logs on port 8501.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
export CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
LOG_DIRECTORY="$(python -c 'import os, yaml; print(yaml.safe_load(open(os.environ["CONFIG_FILE"]))["training"]["tensorboard_directory"])')"
exec tensorboard --logdir "${LOG_DIRECTORY}" --port 8501
