#!/usr/bin/env bash
set -euo pipefail
WORK=/data/mmw/clm_disk/clm-mmw-20261020
CODE=/home/mmw/tls/clm
cd "$CODE"
test "$(findmnt -rn -T "$WORK" -o SOURCE)" = /dev/sda1
test -f "$WORK/runs/bootstrap/DONE"
mkdir -p "$WORK/runs/migration"
exec 9>"$WORK/runs/migration/validation.lock"
flock -n 9 || exit 73
test ! -f "$WORK/runs/migration/DONE"
trap 'echo $? > "$WORK/runs/migration/exit_code"' EXIT
date -Is > "$WORK/runs/migration/STARTED"
exec >"$WORK/runs/migration/validation.log" 2>&1
test "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 256
export CUDA_VISIBLE_DEVICES=0
bash "$CODE/python" "$CODE/scripts/verify_migration.py"
EXPERIMENT="$CODE/experiments/first-memory-fast-refine"
test ! -e "$EXPERIMENT/evaluation"
mkdir "$WORK/runs/migration/evaluation"
ln -s "$WORK/runs/migration/evaluation" "$EXPERIMENT/evaluation"
bash "$CODE/python" "$EXPERIMENT/check_fast.py"
bash "$CODE/python" "$EXPERIMENT/evaluate_fast.py"
date -Is > "$WORK/runs/migration/DONE"
