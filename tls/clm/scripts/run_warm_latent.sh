#!/usr/bin/env bash
set -euo pipefail
cd /home/mmw/tls/clm
RUN=/data/mmw/clm_disk/clm-mmw-20261020/runs/warm-latent-benchmark
mkdir -p "$RUN"
test "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 256
date -Is > "$RUN/LAUNCHED"
trap 'echo $? > "$RUN/exit_code"' EXIT
exec >"$RUN/benchmark.log" 2>&1
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
bash python scripts/benchmark_warm_latent.py
