#!/usr/bin/env bash
# Start the configured fine-tuning run from the repository root.
# Edit config.yaml for training settings. Override GPU_IDS for a one-off run.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f "config.yaml" ]]; then
  echo "config.yaml is missing from ${PROJECT_ROOT}" >&2
  exit 1
fi
export CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
MODEL_DIRECTORY="${MODEL_DIR:-$(python -c 'import os, yaml; print(yaml.safe_load(open(os.environ["CONFIG_FILE"]))["model"]["directory"])')}"
TRAIN_FILE_PATH="${TRAIN_FILE:-$(python -c 'import os, yaml; print(yaml.safe_load(open(os.environ["CONFIG_FILE"]))["data"]["train_file"])')}"
VALIDATION_FILE_PATH="${VAL_FILE:-$(python -c 'import os, yaml; print(yaml.safe_load(open(os.environ["CONFIG_FILE"]))["data"]["validation_file"])')}"

if [[ ! -f "${MODEL_DIRECTORY}/config.json" ]]; then
  echo "Base model is missing at ${MODEL_DIRECTORY}. Run: python model_downloader/download.py" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_FILE_PATH}" || ! -f "${VALIDATION_FILE_PATH}" ]]; then
  echo "Training/validation JSONL files are missing. See the dataset section in README.md." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
exec python train.py
