#!/bin/bash

NOTEBOOK="run-models.ipynb"
OUTPUT_NOTEBOOK="run-models-papermill-output.ipynb"

# change to the project root directory
cd "$(dirname "$0")/../notebooks" || exit 1

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
        break
    else
        echo "Notebook crashed (exit code $EXIT_CODE). Restarting kernel and re-running..."
        sleep 10
    fi
done