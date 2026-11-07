#!/bin/sh
set -eu
M3_PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Default standalone M3; explicit historical weight calls remain explicit.
if [ "${1-}" = "-m" ] && [ "${2-}" = "m3_filament" ] && [ "${3-}" = "predict" ]; then
  M3_EXPLICIT_WEIGHTS=0
  for M3_ARG in "$@"; do
    case "$M3_ARG" in --weights|--weights=*) M3_EXPLICIT_WEIGHTS=1 ;; esac
  done
  if [ "$M3_EXPLICIT_WEIGHTS" = "0" ]; then
    shift 3
    exec "$M3_PROJECT_DIR/predict.sh" "$@"
  fi
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="$M3_PROJECT_DIR/src"
export YOLO_CONFIG_DIR="$M3_PROJECT_DIR/.cache/ultralytics"
export MPLCONFIGDIR="$M3_PROJECT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME="$M3_PROJECT_DIR/.cache"
export PIP_CACHE_DIR="$M3_PROJECT_DIR/.cache/pip"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
cd "$M3_PROJECT_DIR"
exec "$M3_PROJECT_DIR/.env/bin/python" "$@"
