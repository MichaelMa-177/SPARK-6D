#!/usr/bin/env bash
set -euo pipefail
cd /home/mmw/tls/clm
RUN=/data/mmw/clm_disk/clm-mmw-20261020/runs/sam3d-migration
mkdir -p "$RUN"
date -Is > "$RUN/LAUNCHED"
trap 'echo $? > "$RUN/exit_code"' EXIT
exec >"$RUN/download.log" 2>&1
/usr/bin/python3 -Bu scripts/download_sam3d_weights.py
