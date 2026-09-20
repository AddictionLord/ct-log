#!/bin/bash
# Keep a KWP training run alive across crashes, resuming from its last epoch.
#
# Restarts only on abnormal exit. A run that prints its completion marker is
# done and is never relaunched. Repeated fast failures abort rather than
# spinning, so a genuine bug surfaces instead of looping.
#
# Usage: bash scripts/supervise_kwp.sh <config> <run_name> <log_dir> [max_restarts]

set -u
cd /home/mary/code/ct-log
set -a
source .env 2>/dev/null
set +a

CONFIG="${1:?config required}"
RUN_NAME="${2:?run name required}"
LOG_DIR="${3:?log dir required}"
MAX_RESTARTS="${4:-20}"
MIN_ALIVE_SECONDS=120

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run.log"
SUP_LOG="$LOG_DIR/supervisor.log"

restarts=0
while true; do
    if grep -q "Best foreground mean IoU" "$LOG" 2>/dev/null; then
        echo "$(date '+%F %T') run already complete; nothing to do" >> "$SUP_LOG"
        break
    fi

    started=$(date +%s)
    echo "$(date '+%F %T') starting (restart #$restarts)" >> "$SUP_LOG"

    # -u: conda run buffers stdout when it is not a TTY, which leaves the log
    # empty for long stretches and makes the run impossible to monitor.
    conda run -n ct-log --no-capture-output python -u -m src.train_kwp \
        --config "$CONFIG" --run_name "$RUN_NAME" --local_log_dir "$LOG_DIR" \
        >> "$LOG" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - started ))

    if grep -q "Best foreground mean IoU" "$LOG" 2>/dev/null; then
        echo "$(date '+%F %T') completed normally (rc=$rc)" >> "$SUP_LOG"
        break
    fi

    restarts=$(( restarts + 1 ))
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "$(date '+%F %T') giving up after $restarts restarts (rc=$rc)" >> "$SUP_LOG"
        break
    fi
    if [ "$elapsed" -lt "$MIN_ALIVE_SECONDS" ]; then
        echo "$(date '+%F %T') died after ${elapsed}s (rc=$rc) - too fast, likely a real bug; aborting" >> "$SUP_LOG"
        break
    fi

    echo "$(date '+%F %T') died after ${elapsed}s (rc=$rc); resuming in 30s" >> "$SUP_LOG"
    sleep 30
done
