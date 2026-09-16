#!/usr/bin/env bash
# Package code + shipped models + results into a single tarball for the VirtualBox machine.
#   bash scripts/make_release.sh [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-..}
NAME=drl_nav_standalone_$(date +%Y%m%d)
STAMP=$(date -u +"%Y-%m-%d %H:%M UTC")
echo "release built $STAMP" > RELEASE_INFO.txt
python3 --version >> RELEASE_INFO.txt 2>&1 || true
python3 -c "import stable_baselines3 as s, torch, gymnasium as g; print('sb3', s.__version__, 'torch', torch.__version__, 'gymnasium', g.__version__)" >> RELEASE_INFO.txt 2>&1 || true
tar --exclude='__pycache__' --exclude='smoke_runs' --exclude='runs_local' --exclude='*/checkpoints' --exclude='*/tb' \
    --exclude='*.gif.tmp' -czf "$OUT/$NAME.tgz" \
    -C .. drl_nav_standalone
ls -la "$OUT/$NAME.tgz"
echo "unpack with: tar xzf $NAME.tgz"
