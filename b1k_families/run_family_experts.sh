#!/usr/bin/env bash
# Unattended super-family expert orchestrator for BEHAVIOR-1K pi0.5 fine-tunes.
#
# Two-stage, STAR warm-start topology (NOT a chain):
#   Stage 1  pi05_b1k_backbone_foundation  <- warm-start from wave8 checkpoint
#   Stage 2  the four experts (F4,F3,F2,F1) each <- warm-start directly from the
#            completed backbone_foundation checkpoint.
# Each expert is an independent branch -- never chained off another expert, so
# there is no catastrophic forgetting / cross-family skill corruption.
#
# Idempotent + crash-safe: every iteration re-derives state from disk (checkpoint
# dirs, the family ledger, the lerobot cache) -- never from this script's own
# memory. Safe to kill and re-run later; a flock on run_families.lock makes a
# concurrent second invocation exit immediately.
#
# Per family, in order:
#   1. next_family_info.py  -> which family to run (or DONE)
#   2. preflight.py         -> cwd/lerobot/warm-start/VRAM/disk checks; computes
#                              XLA_PYTHON_CLIENT_MEM_FRACTION from free VRAM.
#                              Aborts (does not launch) if VRAM is below the floor.
#   3. ensure_lerobot.py    -> lerobot 0.4.4 guard before touching data.
#   4. stage_family.py      -> task-filtered demo staging (demos 90-99), retries.
#   5. attach-or-launch     -> scripts/train.py $CONFIG --resume --no-wandb-enabled,
#                              bounded crash-retries; completion verified from disk
#                              (family_status --check), never train.py's exit code.
#   6. upload_family.py     -> non-fatal (WARN and keep local on failure).
#   7. NO auto-delete       -> checkpoints persist until you run
#                              cleanup_family.py --yes (after confirming uploads).
# Repeat until DONE.

set -uo pipefail  # deliberately NOT -e: individual steps handle their own failures

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAM_DIR="$OPENPI_ROOT/b1k_families"
PY="$OPENPI_ROOT/.venv/bin/python"
LOCK="$FAM_DIR/run_families.lock"
LOG="$FAM_DIR/run_families.log"
STAGE_WORKERS="${B1K_STAGE_WORKERS:-16}"
MAX_LAUNCH_RETRIES=3
MAX_STAGE_RETRIES=3

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# ---- single-instance lock -----------------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
  log "another run_family_experts.sh is already holding the lock -- exiting cleanly (not an error)"
  exit 0
fi

cd "$OPENPI_ROOT"
log "=== run_family_experts.sh starting (pid $$) ==="

# ---- gate: families only AFTER the wave chain completes --------------------------------
# The full-FT restart trains the wave chain (wave1..wave8) first, then the super-family
# STAR pipeline (backbone_foundation -> F1..F4) warm-starts from the wave8 checkpoint. So
# families must never start while waves are still pending -- their warm-start source does
# not exist yet, and run_waves.sh owns the GPUs. If any wave remains, exit cleanly; re-run
# this script (or have the supervisor) again after run_waves.sh finishes.
next_wave="$("$PY" "$OPENPI_ROOT/b1k_waves/next_wave_info.py")"
nw_rc=$?
if [ "$nw_rc" -ne 0 ]; then
  log "FATAL: could not query wave status (next_wave_info.py exit $nw_rc) to gate family start."
  exit 1
fi
if [ "$next_wave" != "DONE" ]; then
  log "wave chain not yet complete (next wave: ${next_wave:-unknown}). Families must wait for"
  log "  run_waves.sh to finish ALL waves first. Exiting cleanly (nothing to do now)."
  exit 0
fi
log "all waves complete -- proceeding with super-family STAR training."

while true; do
  info="$("$PY" "$FAM_DIR/next_family_info.py")"
  status=$?
  if [ "$status" -ne 0 ]; then
    log "next_family_info.py failed (exit $status) -- cannot determine what to run:"
    echo "$info" | tee -a "$LOG"
    exit 1
  fi
  if [ "$info" = "DONE" ]; then
    log "all families complete. nothing to do."
    exit 0
  fi
  # info is shell-sourceable: NAME=... CONFIG=... TASKS=... LO=... HI=... STEPS=... WARM=...
  eval "$info"
  log "next family: $NAME  config=$CONFIG  ntasks=$(echo "$TASKS" | tr ',' '\n' | wc -l)  steps=$STEPS"
  log "warm-start: $WARM"

  # ---- 2. preflight: resource + integrity checks, computes mem fraction ------------
  pf_out="$("$PY" "$FAM_DIR/preflight.py" --warm-start "$WARM" 2>&1)"
  pf_rc=$?
  echo "$pf_out" | tee -a "$LOG"
  if [ "$pf_rc" -ne 0 ]; then
    log "FATAL: preflight failed for $NAME. Not launching (fix VRAM/disk/lerobot/warm-start, then re-run)."
    exit 1
  fi
  FRACTION="$(printf '%s\n' "$pf_out" | sed -n 's/^FRACTION=//p' | tr -d '[:space:]')"
  FRACTION="${FRACTION:-0.72}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="$FRACTION"
  log "XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION"

  # ---- 3. lerobot guard -------------------------------------------------------------
  if ! "$PY" "$OPENPI_ROOT/b1k_waves/ensure_lerobot.py" >>"$LOG" 2>&1; then
    log "FATAL: could not fix lerobot version. Not launching training on a broken env."
    exit 1
  fi

  # ---- 4. stage this family's task-filtered media, with retries ---------------------
  staged=0
  for attempt in $(seq 1 "$MAX_STAGE_RETRIES"); do
    if "$PY" "$FAM_DIR/stage_family.py" --tasks "$TASKS" --lo "$LO" --hi "$HI" \
         --workers "$STAGE_WORKERS" >>"$LOG" 2>&1; then
      staged=1
      break
    fi
    log "staging attempt $attempt/$MAX_STAGE_RETRIES for $NAME left files missing -- retrying"
    sleep 30
  done
  if [ "$staged" -ne 1 ]; then
    log "FATAL: could not fully stage $NAME after $MAX_STAGE_RETRIES attempts. Stopping. Families must not train on partial data."
    exit 1
  fi

  # ---- 5. attach-or-launch -----------------------------------------------------------
  existing_pid="$(pgrep -f "scripts/train\.py $CONFIG( |$)" | head -1 || true)"
  if [ -n "$existing_pid" ]; then
    log "$CONFIG already running as pid $existing_pid -- waiting on it instead of launching a new one"
    while kill -0 "$existing_pid" 2>/dev/null; do
      sleep 30
    done
    log "pid $existing_pid exited"
  fi

  launched_ok=0
  for try in $(seq 1 "$MAX_LAUNCH_RETRIES"); do
    if "$PY" "$FAM_DIR/family_status.py" --check "$NAME" >>"$LOG" 2>&1; then
      log "$NAME already COMPLETE on disk -- skipping launch"
      launched_ok=1
      break
    fi
    log "launching training for $NAME (attempt $try/$MAX_LAUNCH_RETRIES)"
    "$PY" scripts/train.py "$CONFIG" --resume --no-wandb-enabled \
      >>"$FAM_DIR/train_${NAME}.log" 2>&1
    rc=$?
    if "$PY" "$FAM_DIR/family_status.py" --check "$NAME" >>"$LOG" 2>&1; then
      log "$NAME verified COMPLETE on disk (train.py exit code was $rc)"
      launched_ok=1
      break
    fi
    log "$NAME not complete after train.py exited $rc (attempt $try/$MAX_LAUNCH_RETRIES)."
    log "  tail of train_${NAME}.log:"
    tail -n 20 "$FAM_DIR/train_${NAME}.log" | tee -a "$LOG"
    sleep 30
  done

  if [ "$launched_ok" -ne 1 ]; then
    log "FATAL: $NAME did not complete after $MAX_LAUNCH_RETRIES launch attempts. Stopping."
    log "  Fix the root cause (see train_${NAME}.log), then just re-run this script --"
    log "  it will resume $NAME from its last local checkpoint, not restart it."
    exit 1
  fi

  # ---- 6. upload (non-fatal; keep local on failure) ----------------------------------
  if "$PY" "$FAM_DIR/upload_family.py" "$NAME" >>"$LOG" 2>&1; then
    log "$NAME checkpoint uploaded to 0Corvid0/pi05-b1k-families/${NAME}_100ep"
  else
    log "WARN: upload_family.py failed for $NAME (see $LOG) -- keeping local; can re-upload later."
  fi

  log "=== $NAME complete. NOT deleting local checkpoint; run cleanup_family.py --yes only"
  log "    after you confirm uploads (optionally with --require-upload). ==="
done
