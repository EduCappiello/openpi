#!/usr/bin/env bash
# Unattended wave-training orchestrator for the BEHAVIOR-1K pi0.5 fine-tune.
#
# Run this every time you log in (or from cron @reboot, or a systemd unit -- see
# WAVE_TRAINING.md). It is idempotent and crash-safe: every step re-derives what to
# do from on-disk state (checkpoints, the episode ledger, files present under the
# lerobot cache), never from its own prior run's memory. Killing it at any point --
# API disconnect, `kill`, host reboot -- and re-running it later just picks up where
# it left off. It does NOT depend on any assistant process staying alive.
#
# What it does, per wave, in order:
#   0. Ensure arm0_monolithic params exist locally (idempotent hf download) -- the
#      pretrained warm-start base that wave1 inherits from (full-FT clean-slate re-run).
#   1. Guard the lerobot version (the one real outage this campaign hit).
#   2. Reclaim media not needed by any incomplete wave (frees disk before staging).
#   3. Stage the next wave's demo data.
#   4. Compute warm-start source: local previous-wave checkpoint. With the empty
#      REMOTE_COMPLETE_WAVES + config default chain, wave N inherits wave N-1's local
#      params and wave1 inherits the (fetched) arm0 params -- no hf:// fallback needed.
#   5. Launch/resume training in the FOREGROUND so this script blocks until that
#      wave's process exits, then verifies completion from the checkpoint dir before
#      advancing -- an exit code alone is not trusted.
#   6. Upload completed checkpoint to the gated repo (0Corvid0/pi05-b1k-waves).
#      The old org repo IntelligentDecisionLab/pi05-b1k-monolithic-model is private and
#      has hit its private-storage limit, so uploads go to the public+gated repo instead.
#   7. Prune older local wave checkpoints (frees ~13 GB each; kept current for fallback).
#   8. Repeat until wave_status.py reports no incomplete waves.
#
# Safe to run twice at once: a flock on run_waves.lock makes a second invocation
# exit immediately instead of racing the first.

set -uo pipefail  # deliberately NOT -e: individual steps handle their own failures

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAVES_DIR="$OPENPI_ROOT/b1k_waves"
PY="$OPENPI_ROOT/.venv/bin/python"
LOCK="$WAVES_DIR/run_waves.lock"
LOG="$WAVES_DIR/run_waves.log"
STAGE_WORKERS="${B1K_STAGE_WORKERS:-16}"
MAX_LAUNCH_RETRIES=3
REPO_ID="0Corvid0/pi05-b1k-waves"

# GPU memory: FULL fine-tuning with fsdp_devices=4 shards the ~2.4B model + fp32 Adam
# states across 4x H100 80 GB. Preallocate 0.75 (~60 GB/GPU) for training so the XLA
# pool fits alongside any model server; lower the floor (e.g. 0.60) only if needed.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# ---- single-instance lock -----------------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
  log "another run_waves.sh is already holding the lock -- exiting cleanly (not an error)"
  exit 0
fi
# lock auto-releases when this process (fd 9) exits, including on kill -9 of the shell

cd "$OPENPI_ROOT"
log "=== run_waves.sh starting (pid $$) ==="

# ---- ensure arm0 warm-start base is present (idempotent) -------------------------------
# Wave1 inherits from arm0_monolithic (config default chain). Its params are NOT shipped
# in this repo and, on a clean-slate re-run, are gone locally -- fetch them via the HF
# download CLI (fsspec hf:// directory downloads are unreliable on this pod; the CLI is
# the working method, see WAVE_TRAINING.md "checkpoint download trap"). No-op if present.
ARM0_DIR="./outputs/checkpoints/pi05_b1k/arm0_monolithic/49999/params"
# arm0 params live under the monolithic repo's arm0_monolithic_30ep/ folder. The older
# separate "bucket" repo (pi05-b1k-arm0-monolithic-bucket-30ep) no longer exists / 404s.
# Download with the `snapshot_download()` FUNCTION via python -c: proven and robust on this
# pod. (The `hf download` CLI also works, but `python -m huggingface_hub.snapshot_download`
# / `python -m huggingface_hub download` are NOT runnable modules -- no __main__ -- and fsspec
# hf:// directory downloads silently fail here, so avoid both of those.)
ARM0_HF_REPO="IntelligentDecisionLab/pi05-b1k-monolithic-model"
ARM0_HF_FOLDER="arm0_monolithic_30ep"
ARM0_STAGE="./outputs/checkpoints/_arm0_stage"
fetch_arm0() {
  if [ -d "$(pwd)/$ARM0_DIR" ]; then
    log "arm0 warm-start base already present at $ARM0_DIR -- skipping fetch"
    return 0
  fi
  log "arm0 warm-start base missing; downloading params from $ARM0_HF_REPO/$ARM0_HF_FOLDER via snapshot_download"
  rm -rf "$(pwd)/$ARM0_STAGE"
  mkdir -p "$(dirname "$(pwd)/$ARM0_DIR")" "$(pwd)/$ARM0_STAGE"
  if "$PY" -c '
from huggingface_hub import snapshot_download
import pathlib, sys
stage = pathlib.Path(sys.argv[1]).resolve()
snapshot_download(sys.argv[2], allow_patterns=[sys.argv[3] + "/params/*"], local_dir=str(stage))
' "$(pwd)/$ARM0_STAGE" "$ARM0_HF_REPO" "$ARM0_HF_FOLDER" >>"$LOG" 2>&1 \
     && [ -d "$(pwd)/$ARM0_STAGE/$ARM0_HF_FOLDER/params" ] \
     && mv "$(pwd)/$ARM0_STAGE/$ARM0_HF_FOLDER/params" "$(pwd)/$ARM0_DIR"; then
    log "arm0 downloaded to $ARM0_DIR"
  else
    log "WARN: arm0 download failed -- wave1 will fall back to base gemma_2b init. See $LOG."
  fi
  rm -rf "$(pwd)/$ARM0_STAGE"
}
fetch_arm0

# ---- main loop ------------------------------------------------------------------------
while true; do
  info="$("$PY" "$WAVES_DIR/next_wave_info.py")"
  status=$?
  if [ "$status" -ne 0 ]; then
    log "next_wave_info.py failed (exit $status) -- cannot determine what to run:"
    echo "$info" | tee -a "$LOG"
    exit 1
  fi
  if [ "$info" = "DONE" ]; then
    log "all queued waves complete. nothing to do."
    exit 0
  fi
  # info is shell-sourceable: NAME=... CONFIG=... LO=... HI=... STEPS=...
  eval "$info"
  log "next wave: $NAME  demos[$LO,$HI)  config=$CONFIG  steps=$STEPS"

  # ---- warm-start source (local config default chain) ------------------------------
  # _make_wave_configs() already wires each wave to warm from the previous wave's local
  # checkpoint (wave1 -> the arm0 base fetched above). Unset any override so the chain
  # is authoritative; local-first means no hf:// download is attempted at train time.
  unset B1K_WARM_START_PARAMS
  log "warm-start: config default chain (wave1->arm0, wave N->wave N-1 local params)"

  # 1. lerobot version guard -----------------------------------------------------------
  if ! "$PY" "$WAVES_DIR/ensure_lerobot.py" >>"$LOG" 2>&1; then
    log "FATAL: could not fix lerobot version. Not launching training on a broken env."
    exit 1
  fi

  # 2. reclaim disk before staging -----------------------------------------------------
  "$PY" "$WAVES_DIR/reclaim.py" >>"$LOG" 2>&1
  log "reclaim done (see $LOG for freed GB)"

  # 3. stage this wave's media, with retries on top of stage_wave.py's own retries ------
  staged=0
  for attempt in 1 2 3; do
    if "$PY" "$WAVES_DIR/stage_wave.py" --lo "$LO" --hi "$HI" --workers "$STAGE_WORKERS" >>"$LOG" 2>&1; then
      staged=1
      break
    fi
    log "staging attempt $attempt/3 for $NAME left files missing -- retrying"
    sleep 30
  done
  if [ "$staged" -ne 1 ]; then
    log "FATAL: could not fully stage $NAME after 3 attempts. Stopping (not skipping -- next"
    log "  run of this script will retry the same wave; waves must not train on partial data)."
    exit 1
  fi

  # 3b. if a trainer for THIS config is already running (e.g. launched manually, or by
  # an earlier invocation of this script that is still going), attach to it instead of
  # launching a second one -- two processes writing the same orbax checkpoint dir can
  # corrupt it, and a second process would also fight the first for GPU memory.
  existing_pid="$(pgrep -f "scripts/train\.py $CONFIG( |$)" | head -1 || true)"
  if [ -n "$existing_pid" ]; then
    log "$CONFIG already running as pid $existing_pid -- waiting on it instead of launching a new one"
    while kill -0 "$existing_pid" 2>/dev/null; do
      sleep 30
    done
    log "pid $existing_pid exited"
  fi

  # 4. launch/resume training, in the foreground, with bounded crash-retries -----------
  launched_ok=0
  for try in $(seq 1 "$MAX_LAUNCH_RETRIES"); do
    if "$PY" "$WAVES_DIR/wave_status.py" --check "$NAME" >>"$LOG" 2>&1; then
      log "$NAME already COMPLETE on disk -- skipping launch"
      launched_ok=1
      break
    fi
    log "launching training for $NAME (attempt $try/$MAX_LAUNCH_RETRIES)"
    # Cap CPU thread pools to the pod's cgroup cpu quota (40 CPUs). OpenBLAS/OMP/JAX
    # otherwise spawn up to 64 threads and die at checkpoint-save with
    # "std::system_error: Resource temporarily unavailable" / "pthread_create failed".
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-40}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-40}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS:-40}" \
    "$PY" scripts/train.py "$CONFIG" --resume --no-wandb-enabled \
      >>"$WAVES_DIR/train_${NAME}.log" 2>&1
    rc=$?
    if "$PY" "$WAVES_DIR/wave_status.py" --check "$NAME" >>"$LOG" 2>&1; then
      log "$NAME verified COMPLETE on disk (train.py exit code was $rc)"
      launched_ok=1
      break
    fi
    log "$NAME not complete after train.py exited $rc (attempt $try/$MAX_LAUNCH_RETRIES)."
    log "  tail of train_${NAME}.log:"
    tail -n 20 "$WAVES_DIR/train_${NAME}.log" | tee -a "$LOG"
    sleep 30
  done

  if [ "$launched_ok" -ne 1 ]; then
    log "FATAL: $NAME did not complete after $MAX_LAUNCH_RETRIES launch attempts. Stopping."
    log "  Fix the root cause (see train_${NAME}.log), then just re-run this script --"
    log "  it will resume $NAME from its last local checkpoint, not restart it."
    exit 1
  fi

  # 5. upload completed checkpoint to unified HF repo ----------------------------------
  uploaded=0
  if "$PY" "$WAVES_DIR/upload_wave.py" "$NAME" >>"$LOG" 2>&1; then
    log "$NAME checkpoint uploaded to $REPO_ID (see $LOG for folder name)"
    uploaded=1
  else
    log "WARN: upload_wave.py failed for $NAME (see $LOG) -- continuing without pruning;"
    log "      the local checkpoint is preserved and can be re-uploaded later."
  fi

  # 6. prune older local wave checkpoints (only after successful upload) ---------------
  # SAFETY GATE: only prune a checkpoint whose wave is recorded in REMOTE_COMPLETE_WAVES
  # (wave_status.py). A wave is safe to delete only once it is verified present in HF;
  # the disk-only completeness check otherwise re-selects a pruned-but-not-uploaded wave
  # and tries to re-train already-consumed demos. Deleting every non-current wave was the
  # root cause of the 2026-08-15 bug that pruned wave3 (uploaded but not yet recorded) and
  # then re-selected wave3. Always keep the current wave regardless.
  if [ "$uploaded" -eq 1 ]; then
    pruned_freed=0
    for d in outputs/checkpoints/pi05_b1k_wave*/; do
      [ ! -d "$d" ] && continue
      base="$(basename "$d")"
      wave="${base#pi05_b1k_}"                # dir basename -> wave ledger name
      [ "$wave" = "$NAME" ] && continue       # keep current wave
      if ! "$PY" -c "import sys;sys.path.insert(0, sys.argv[1]);import wave_status as WS;sys.exit(0 if '$wave' in WS.REMOTE_COMPLETE_WAVES else 1)" "$WAVES_DIR"; then
        log "skipping prune of $base (wave '$wave' not recorded in REMOTE_COMPLETE_WAVES -- keep local fallback)"
        continue
      fi
      sz=$(du -sm "$d" | cut -f1)
      log "pruning old checkpoint $base (${sz} MB, wave '$wave' verified in HF)"
      rm -rf "$d"
      pruned_freed=$((pruned_freed + sz))
    done
    if [ "$pruned_freed" -gt 0 ]; then
      log "pruned ${pruned_freed} MB of older local wave checkpoints (kept $NAME)"
    else
      log "nothing to prune ($NAME was the only prunable pi05_b1k_wave* checkpoint)"
    fi
  fi

  log "=== $NAME complete -- advancing to next wave ==="
done
