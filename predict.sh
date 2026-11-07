#!/bin/sh
set -eu
M3_DEPLOY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH="$M3_DEPLOY_ROOT/src:$M3_DEPLOY_ROOT"
export YOLO_CONFIG_DIR="$M3_DEPLOY_ROOT/.cache/ultralytics"
export MPLCONFIGDIR="$M3_DEPLOY_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$M3_DEPLOY_ROOT/.cache"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd "$M3_DEPLOY_ROOT"
exec "$M3_DEPLOY_ROOT/.team-env/bin/python" "$M3_DEPLOY_ROOT/deployments/predict_default.py" "$@"
