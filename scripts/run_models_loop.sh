#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

echo "scripts/run_models_loop.sh is now a compatibility wrapper."
echo "The resilient iterative runner lives in run_experiments.py."

python3 run_experiments.py run "$@"
