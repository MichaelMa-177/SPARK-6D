#!/usr/bin/env bash
set -euo pipefail
CODE=/home/mmw/tls/clm
WORK=/data/mmw/clm_disk/clm-mmw-20261020
RUN="$WORK/runs/fresh-sam3d-pipeline"
EXP="$CODE/experiments/first-memory-fresh-sam3d"
cd "$CODE"
test "$(findmnt -rn -T "$WORK" -o SOURCE)" = /dev/sda1
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || exit 73
test ! -e "$RUN/DONE"
date -Is > "$RUN/STARTED"
trap 'echo $? > "$RUN/exit_code"' EXIT
exec >"$RUN/pipeline.log" 2>&1
test "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 256
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
if test ! -d "$EXP"; then
    mkdir "$EXP"
    for file in geometry.py data_fast.py fast_model.py train_fast.py evaluate_fast.py check_fast.py; do
        cp "$CODE/experiments/first-memory-fast-refine/$file" "$EXP/$file"
    done
    ln -s "$CODE/experiments/first-memory-fast-refine/inputs" "$EXP/inputs"
    ln -s "$CODE/experiments/first-memory-fast-refine/training" "$EXP/training"
    mkdir "$RUN/priors" "$RUN/evaluation"
    ln -s "$RUN/priors" "$EXP/priors"
    ln -s "$RUN/evaluation" "$EXP/evaluation"
fi
bash "$CODE/python" "$CODE/scripts/prepare_fresh_prior.py"
bash "$CODE/python" "$EXP/evaluate_fast.py"
date -Is > "$RUN/DONE"
