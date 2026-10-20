#!/usr/bin/env bash
set -euo pipefail
cd /home/mmw/tls/clm
WORK=/data/mmw/clm_disk/clm-mmw-20261020
test "$(findmnt -rn -T "$WORK" -o SOURCE)" = /dev/sda1
mkdir -p "$WORK/runs/self-data-download"
date -Is > "$WORK/runs/self-data-download/LAUNCHED"
trap 'echo $? > "$WORK/runs/self-data-download/exit_code"' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec >"$WORK/runs/self-data-download/download.log" 2>&1
/usr/bin/python3 -B scripts/download_self_data.py
