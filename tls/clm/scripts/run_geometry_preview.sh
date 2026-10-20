#!/usr/bin/env bash
set -euo pipefail
cd /home/mmw/tls/clm
RUN=/data/mmw/clm_disk/clm-mmw-20261020/runs/fresh-geometry-preview
mkdir -p "$RUN"
exec 9>"$RUN/export.lock"
flock -n 9 || exit 73
test ! -e "$RUN/DONE"
test "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 256
trap 'echo $? > "$RUN/exit_code"' EXIT
exec >"$RUN/export.log" 2>&1
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
bash python scripts/export_fresh_geometry.py
