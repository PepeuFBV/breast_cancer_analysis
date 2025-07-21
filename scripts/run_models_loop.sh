#!/bin/bash

NOTEBOOK="run-models.ipynb"
OUTPUT_NOTEBOOK="run-models-papermill-output.ipynb"
LOG_FILE="log.txt"

# change to the project root directory
cd "$(dirname "$0")/../notebooks" || exit 1

echo "$(date '+%Y-%m-%d %H:%M:%S') - Script manually started." >> "$LOG_FILE"

if [ ! -f "$NOTEBOOK" ]; then
    echo "Notebook $NOTEBOOK not found."
    exit 1
fi

while true; do 
    echo "Current directory: $(pwd)"
    echo "Running notebook with papermill..."
    papermill "$NOTEBOOK" "$OUTPUT_NOTEBOOK" --log-output
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Notebook finished successfully."
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Notebook finished successfully." >> "$LOG_FILE"
        break
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "Notebook exited with code 1. Not restarting."
        break
    else
        echo "Notebook crashed (exit code $EXIT_CODE). Restarting kernel and re-running..."
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Notebook crashed (exit code $EXIT_CODE). Restarting kernel and re-running..." >> "$LOG_FILE"
        sleep 10
    fi
done