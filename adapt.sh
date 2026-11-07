#!/bin/sh
set -eu
ADAPT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ADAPT_ROOT/src:$ADAPT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DISABLE_SAFETENSORS_CONVERSION=1
export HF_HOME="$ADAPT_ROOT/.cache/team-hf"
export XDG_CACHE_HOME="$ADAPT_ROOT/.cache"
export MPLCONFIGDIR="$ADAPT_ROOT/.cache/matplotlib"
export YOLO_CONFIG_DIR="$ADAPT_ROOT/.cache/ultralytics"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd "$ADAPT_ROOT"
exec "$ADAPT_ROOT/.team-env/bin/python" -m team_adaptation "$@"
