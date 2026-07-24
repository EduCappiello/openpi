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
#   1. Guard the lerobot version (the one real outage this campaign hit: a `uv sync`
#      elsewhere silently downgraded lerobot 0.4.4 -> 0.3.4, which cannot read the
#      v3.0 dataset layout and crashes training on the first launch after).
#   2. Reclaim media not needed by any incomplete wave (frees disk before staging).
#   3. Stage the next wave's media (retries internally; see stage_wave.py).
#   4. Launch/resume training in the FOREGROUND so this script blocks until that
#      wave's process exits, then verifies completion from the checkpoint dir before
#      advancing -- an exit code alone is not trusted.
#   5. Repeat until wave_status.py reports no incomplete waves.
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
  # --resume is safe unconditionally: empty/missing checkpoint dir -> fresh warm start
  # from the previous wave's final params; partial checkpoint dir -> resumes that
  # wave's own progress. Check completion BEFORE each attempt too (cheap, disk-only) --
  # the process we attached to above may have just finished the wave, and re-launching
  # train.py just to discover that costs ~2min of dataset-scan + XLA compile for nothing.
  launched_ok=0
  for try in $(seq 1 "$MAX_LAUNCH_RETRIES"); do
    if "$PY" "$WAVES_DIR/wave_status.py" --check "$NAME" >>"$LOG" 2>&1; then
      log "$NAME already COMPLETE on disk -- skipping launch"
      launched_ok=1
      break
    fi
    log "launching training for $NAME (attempt $try/$MAX_LAUNCH_RETRIES)"
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

  log "=== $NAME complete -- advancing to next wave ==="
done
