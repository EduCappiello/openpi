#!/usr/bin/env bash
# Keep the Arm 0 pi05_b1k run alive across crashes, OOMs and disconnects.
#
# Runs detached (setsid) so it survives the session that started it. It never launches a
# second trainer: if a train.py process is already alive it simply idles.
#
# Restart policy: resume from the newest checkpoint when one exists, otherwise start fresh.
# Rapid consecutive failures back off, so a config error cannot become a hot restart loop
# that repeatedly clobbers checkpoints.

set -uo pipefail

OPENPI=/root/dev/b1k-baselines/baselines/openpi
EXP=arm0_monolithic
FINAL_STEP=50000   # must match num_train_steps in the pi05_b1k config
CKPT_DIR="$OPENPI/outputs/checkpoints/pi05_b1k/$EXP"
LOG_DIR=/root/b1k-logs
TRAIN_LOG="$LOG_DIR/arm0.log"
SUP_LOG="$LOG_DIR/supervisor.log"
STATUS="$LOG_DIR/STATUS.txt"
# v2: the original lock is permanently pinned by the running trainer, which inherited the
# lock fd from the supervisor that spawned it (see 9>&- on the launch below).
LOCK="$LOG_DIR/supervisor.v2.lock"

mkdir -p "$LOG_DIR"

# Single supervisor only.
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another supervisor holds $LOCK; exiting" >&2
    exit 0
fi

log() { echo "[$(date -u '+%F %T')] $*" >> "$SUP_LOG"; }

# Identify the trainer by process *identity*, not by command-line text alone.
# A bare `pgrep -f "train.py pi05_b1k"` also matches any shell (or monitoring command)
# whose own argv happens to contain that string, which made the supervisor believe a dead
# trainer was still alive and silently skip every restart. Requiring comm to be the python
# interpreter excludes zsh/bash wrappers.
trainer_pid() {
    local p comm
    for p in $(pgrep -f "train\.py pi05_b1k" 2>/dev/null); do
        comm=$(cat "/proc/$p/comm" 2>/dev/null) || continue
        case "$comm" in
            python*) echo "$p"; return 0 ;;
        esac
    done
    return 1
}

fails=0
log "supervisor started (pid $$)"

while true; do
    if [ -n "$(trainer_pid)" ]; then
        fails=0
        # train.py's "Step N: loss=..." goes to stdout via tqdm_loggable's pbar.write(),
        # which never reaches the log file. The tqdm progress line on stderr does, so
        # take the step count from there.
        step=$(grep -oE "Progress on: [0-9.]+k?it/" "$TRAIN_LOG" 2>/dev/null | tail -1 \
               | grep -oE "[0-9.]+k?it" | head -1)
        # tqdm.write() prefixes a CR to clear the bar, so the line does NOT start at
        # column 0 -- never anchor this pattern with ^.
        loss=$(grep -oE "Step [0-9]+: .*" "$TRAIN_LOG" 2>/dev/null | tail -1)
        {
            echo "state      : RUNNING (pid $(trainer_pid))"
            echo "last step  : ${step:-<startup>}"
            echo "last metrics: ${loss:-<none yet>}"
            echo "checkpoints: $(ls "$CKPT_DIR" 2>/dev/null | tr '\n' ' ')"
            echo "disk free  : $(df -h / | awk 'NR==2{print $4}')"
            echo "updated    : $(date -u '+%F %T')"
        } > "$STATUS"
        sleep 60 9>&-
        continue
    fi

    # Trainer is gone. Decide whether it finished or died.
    if grep -q "Training completed\|Training finished" "$TRAIN_LOG" 2>/dev/null; then
        log "training reported completion; supervisor exiting"
        echo "state: COMPLETED  updated: $(date -u '+%F %T')" > "$STATUS"
        exit 0
    fi
    # Completion is detected from the final checkpoint on disk, not from log text.
    # log_interval=100 means the last "Step N:" line printed is 49900, so a step-based
    # threshold never fires and a finished run would be restart-looped forever.
    if [ -d "$CKPT_DIR/$FINAL_STEP" ]; then
        log "final checkpoint $FINAL_STEP present; training complete, supervisor exiting"
        echo "state: COMPLETED at step $FINAL_STEP  updated: $(date -u '+%F %T')" > "$STATUS"
        exit 0
    fi
    last_step=$(grep -oE "Progress on: [0-9.]+k?it/" "$TRAIN_LOG" 2>/dev/null | tail -1 \
                | grep -oE "[0-9.]+k?it" | head -1)

    fails=$((fails + 1))
    log "trainer not running (failure #$fails). last step=${last_step:-none}"
    log "last log lines: $(tail -n 3 "$TRAIN_LOG" 2>/dev/null | tr '\n' ' | ')"

    if [ "$fails" -ge 6 ]; then
        log "6 consecutive failures - stopping to avoid a restart loop. Investigate $TRAIN_LOG"
        echo "state: FAILED after $fails restarts. See $SUP_LOG" > "$STATUS"
        exit 1
    fi

    # Back off progressively: 30s, 60s, 120s, 240s, 480s.
    backoff=$((30 * (1 << (fails - 1))))
    log "backing off ${backoff}s before restart"
    sleep "$backoff" 9>&-

    if [ -d "$CKPT_DIR" ] && [ -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
        mode="--resume"
    else
        mode="--overwrite"
    fi
    log "restarting trainer with $mode"
    mv -f "$TRAIN_LOG" "$TRAIN_LOG.$(date -u '+%Y%m%dT%H%M%SZ')" 2>/dev/null
    cd "$OPENPI" || exit 1
    # PYTHONUNBUFFERED so the per-log_interval "Step N: loss=..., grad_norm=..." lines are
    # flushed to the log instead of sitting in a block buffer. Without it the run is alive
    # but its loss is invisible until the process exits.
    # 9>&- closes the inherited flock fd in the child. Without it the trainer holds the
    # supervisor's lock for its entire lifetime, so if this supervisor ever dies no
    # replacement can acquire the lock and the run silently becomes unsupervised.
    setsid nohup env PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/train.py pi05_b1k \
        --exp-name="$EXP" $mode --no-wandb-enabled >> "$TRAIN_LOG" 2>&1 9>&- &
    log "relaunched trainer pid $!"
    sleep 120 9>&- # give it time to come up before the next health check
done
