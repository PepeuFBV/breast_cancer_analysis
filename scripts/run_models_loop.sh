#!/bin/bash

LOG_FILE="artifacts/runs/logs/train-loop.log"

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$(dirname "$LOG_FILE")"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Script manually started." >> "$LOG_FILE"

while true; do
    echo "Current directory: $(pwd)"
    echo "Running train.py..."
    python3 train.py "$@"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training finished successfully."
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Training finished successfully." >> "$LOG_FILE"
        break
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "Training exited with code 1. Not restarting."
        break
    else
        echo "Training crashed (exit code $EXIT_CODE). Restarting after cooldown..."
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Training crashed (exit code $EXIT_CODE). Restarting." >> "$LOG_FILE"
        sleep 10
    fi
done
