#!/bin/sh
# Rebuild a separate prefix. Never modifies .env/base. Does not download model weights.
set -eu
TEAM_SETUP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TEAM_CONDA_BIN=${TEAM_CONDA_BIN:-conda}
TEAM_SETUP_PREFIX=${TEAM_SETUP_PREFIX:-"$TEAM_SETUP_ROOT/.team-env"}
case "$TEAM_SETUP_PREFIX" in "$TEAM_SETUP_ROOT"/.team-env*) ;; *) echo 'Prefix must be a dedicated .team-env* inside this project' >&2;exit 2;; esac
if [ -e "$TEAM_SETUP_PREFIX" ]; then echo 'Target already exists; use a new TEAM_SETUP_PREFIX ending in .team-env*' >&2;exit 2;fi
export CONDA_REGISTER_ENVS=false
export CONDA_PKGS_DIRS="$TEAM_SETUP_ROOT/.cache/team-conda-pkgs"
export PIP_CACHE_DIR="$TEAM_SETUP_ROOT/.cache/team-pip"
"$TEAM_CONDA_BIN" create --prefix "$TEAM_SETUP_PREFIX" --no-default-packages python=3.11.16 pip -y
"$TEAM_SETUP_PREFIX/bin/python" -m pip install -r "$TEAM_SETUP_ROOT/team/requirements-team-lock.txt"
"$TEAM_SETUP_PREFIX/bin/python" -m pip check
