#!/bin/sh
set -eu
TEAM_PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH="$TEAM_PROJECT_DIR/src:$TEAM_PROJECT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DISABLE_SAFETENSORS_CONVERSION=1
export HF_HOME="$TEAM_PROJECT_DIR/.cache/team-hf"
export XDG_CACHE_HOME="$TEAM_PROJECT_DIR/.cache"
export MPLCONFIGDIR="$TEAM_PROJECT_DIR/.cache/matplotlib"
export YOLO_CONFIG_DIR="$TEAM_PROJECT_DIR/.cache/ultralytics"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd "$TEAM_PROJECT_DIR"
exec "$TEAM_PROJECT_DIR/.team-env/bin/python" -m team_pipeline "$@"
