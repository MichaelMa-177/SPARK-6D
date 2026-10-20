#!/usr/bin/env bash
set -euo pipefail
cd /home/mmw/tls/clm
WORK=/data/mmw/clm_disk/clm-mmw-20261020
test "$(findmnt -rn -T "$WORK" -o SOURCE)" = /dev/sda1
mkdir -p "$WORK/runs/bootstrap"
trap 'echo $? > "$WORK/runs/bootstrap/exit_code"' EXIT
exec >"$WORK/runs/bootstrap/setup.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 /usr/bin/python3 -B scripts/bootstrap_env.py
