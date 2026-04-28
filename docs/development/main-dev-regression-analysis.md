# Main vs Dev Regression Analysis

## What changed

`main` ran experiments from `notebooks/run-models.ipynb` through
`scripts/run_models_loop.sh`. The notebook trained models inline and, on
TensorFlow `ResourceExhaustedError`, called `os._exit(1)` so Papermill could
restart the whole process.

`dev` replaced that notebook-first flow with an in-process Python pipeline and
long-lived iterative runner centered on:

- `pipeline/experiments/runner.py`
- `pipeline/train/runner.py`
- `pipeline/train/models.py`
- `pipeline/train/preprocessing.py`
- `pipeline/data/dataset.py`
- `pipeline/utils/gpu_env.py`
- `run_experiments.py`

## Why the long-run failures are likely lifecycle issues

The old flow recovered from TensorFlow/Keras memory growth by restarting the
Python process. The refactored `dev` runner now executes many experiment
combinations inside one process, while repeatedly loading image arrays,
expanding grayscale inputs to RGB for transfer-learning models, building Keras
models, and predicting on evaluation splits.

That makes failures around the 8th or 9th combination consistent with resource
retention across tasks and folds rather than queue-state logic:

- TensorFlow/Keras state can survive across runs in the same process.
- Large NumPy arrays can coexist longer than necessary during fit/predict.
- Cross-validation repeats the same allocations multiple times per task.
- GPU setup is no longer isolated to a fresh process boundary.

## Fixes and tests added on this branch

This branch adds shared memory cleanup helpers, deterministic per-task and
per-fold cleanup in the training runner, clearer CPU/GPU runtime selection and
reporting, long-run regression coverage, and a tiny real-training validation
path that exercises more than 9 combinations without running the full
multi-hour experiment grid.
