#!/bin/sh
set -eu
M3_PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$M3_PROJECT_DIR"
if [ "${1:-}" = '--check' ]; then
  ./run.sh -c 'import sys, site; assert sys.version_info[:2] == (3,11); assert not site.ENABLE_USER_SITE; print(sys.executable)'
  ./run.sh -m pip check
  exit 0
fi
if [ ! -x .env/bin/python ]; then
  M3_CONDA_BIN=${M3_CONDA_BIN:-$(command -v conda)}
  mkdir -p .cache/conda-pkgs
  CONDA_REGISTER_ENVS=false CONDA_PKGS_DIRS="$M3_PROJECT_DIR/.cache/conda-pkgs" \
    "$M3_CONDA_BIN" create --copy --no-default-packages -p "$M3_PROJECT_DIR/.env" python=3.11.16 pip -y
fi
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$M3_PROJECT_DIR/.cache/pip"
.env/bin/python -m pip install --find-links .cache/bootstrap-wheels -r requirements-lock.txt
./setup.sh --check
