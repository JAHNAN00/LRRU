#!/usr/bin/env bash

set -u

ROOT="/home/an/Desktop/LRRU"
CONDA_SH="/home/an/miniconda3/etc/profile.d/conda.sh"
LOG_DIR="$ROOT/run_logs"
LOG_FILE="$LOG_DIR/mini_37epoch_best_resume_watch.log"
STATUS_FILE="$LOG_DIR/mini_37epoch_best_resume_watch.status"
PID_FILE="$LOG_DIR/mini_37epoch_best_resume_watch.pid"
CONFIG="train_lrru_mini_37epoch_best_resume_kitti.yml"
MIN_FREE_GB="15"
CHECK_INTERVAL="30"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        printf '已有训练进程在运行: %s\n' "$OLD_PID"
        exit 1
    fi
fi

if [[ ! -f "$CONDA_SH" ]]; then
    printf 'missing conda init script: %s\n' "$CONDA_SH"
    exit 1
fi

bash -lc "source '$CONDA_SH' && conda activate LRRU && cd '$ROOT' && exec python train_apex.py -c '$CONFIG'" >> "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$PID_FILE"
printf 'started pid=%s log=%s\n' "$TRAIN_PID" "$LOG_FILE" > "$STATUS_FILE"

cleanup() {
    rm -f "$PID_FILE"
}

trap cleanup EXIT

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    FREE_GB="$(df -BG "$ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"

    if [[ -n "$FREE_GB" ]] && (( FREE_GB < MIN_FREE_GB )); then
        printf 'stop: free disk %sG below threshold %sG\n' "$FREE_GB" "$MIN_FREE_GB" | tee -a "$STATUS_FILE"
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        sleep 10
        kill -KILL "$TRAIN_PID" 2>/dev/null || true
        exit 2
    fi

    if grep -q "No space left on device" "$LOG_FILE"; then
        printf 'stop: detected no space left on device\n' | tee -a "$STATUS_FILE"
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        sleep 10
        kill -KILL "$TRAIN_PID" 2>/dev/null || true
        exit 3
    fi

    sleep "$CHECK_INTERVAL"
done

wait "$TRAIN_PID"
EXIT_CODE=$?
printf 'finished exit_code=%s\n' "$EXIT_CODE" | tee -a "$STATUS_FILE"
exit "$EXIT_CODE"
