#!/bin/sh
set -eu
RANDOM_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RANDOM_ROOT/src:$RANDOM_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DISABLE_SAFETENSORS_CONVERSION=1
export HF_HOME="$RANDOM_ROOT/.cache/team-hf"
export XDG_CACHE_HOME="$RANDOM_ROOT/.cache"
export MPLCONFIGDIR="$RANDOM_ROOT/.cache/matplotlib"
export YOLO_CONFIG_DIR="$RANDOM_ROOT/.cache/ultralytics"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd "$RANDOM_ROOT"
exec "$RANDOM_ROOT/.team-env/bin/python" -m random_validation "$@"
