#!/usr/bin/env bash
# 2-GPU smoke test: ~100 training steps on the wave8 config, warm-started from
# the local arm0 base, checkpoint written then pruned. No uploads, no ledger
# mutation. Run from the openpi repo root.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.60}"
# wave8's config-chain default warm-start (wave7's local params) does not exist on a
# clean machine; warm-start from the fetched arm0 base for the smoke run.
export B1K_WARM_START_PARAMS="${B1K_WARM_START_PARAMS:-./outputs/checkpoints/pi05_b1k/arm0_monolithic/49999/params}"
# Per-wave subset HF dataset cache (~5 GB/wave) -> big volume, not "/"
export B1K_HF_DATASET_CACHE="${B1K_HF_DATASET_CACHE:-/tmp/hf-dataset-cache}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-20}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"

.venv/bin/python b1k_waves/ensure_lerobot.py || { echo "FATAL: lerobot guard failed"; exit 1; }

.venv/bin/python scripts/train.py pi05_b1k_wave8_d90_100 \
  --overwrite \
  --no-wandb-enabled \
  --num-train-steps 100 \
  --save-interval 100 \
  --log-interval 20 \
  2>&1 | tee smoke_test.log

rc=${PIPESTATUS[0]}
echo "train.py exit code: $rc"
ls -d outputs/checkpoints/pi05_b1k_wave8_d90_100/wave8_d90_100/* 2>/dev/null || echo "NO CHECKPOINT DIR"
exit $rc
