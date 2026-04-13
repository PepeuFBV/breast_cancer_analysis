# Breast Cancer Analysis

This repository contains a reproducible Python pipeline for mammogram classification experiments based on the INbreast dataset. The preprocessing, training, and evaluation logic now lives in reusable Python modules under `pipeline/`, while the notebooks remain available as exploratory and reporting interfaces.

The repository also includes the paper source in [`article/main.tex`](article/main.tex) and the compiled PDF in [`article/main.pdf`](article/main.pdf).

## What Changed

The project used to be notebook-first. The main workflow is now organized around Python entrypoints:

- `preprocess.py` builds processed images and train/test splits
- `train.py` runs the experiment grid and stores model artifacts
- `run_experiments.py` executes the same grid through a resumable iterative runner
- `experiment_dashboard.py` provides a local Run/Stop dashboard on top of the same runner
- `evaluate.py` aggregates run outputs into a final report

The notebooks in [`notebooks/`](notebooks) now consume those modules instead of owning the full pipeline logic.

## Project Layout

```text
pipeline/
  data/
  experiments/
  train/
  evaluate/
  utils/
artifacts/
  experiments/
  processed/
  runs/
  reports/
data/
  INbreast Release 1.0/
notebooks/
preprocess.py
train.py
run_experiments.py
experiment_dashboard.py
evaluate.py
```

- converts INbreast DICOM images to normalized PNG files,
- applies leakage-aware splitting, training-only augmentation, and a balanced sampling strategy,
- trains multiple model architectures on different preprocessing variants,
- stores per-run histories and predictions, and
- aggregates results into a final CSV for comparison.

The code is built around **INbreast Release 1.0**.

Expected raw dataset layout:

```text
data/INbreast Release 1.0/
  INbreast.csv
  AllDICOMs/*.dcm
```

Valid BI-RADS labels handled by the pipeline are:

- `1`
- `2`
- `3`
- `4a`
- `4b`
- `4c`
- `5`
- `6`

The dataset is not tracked by Git and must be placed locally in the path above.

## Installation

Create a local environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install is recommended so notebook imports like `from pipeline...` work cleanly.

## Code Quality Checks

With the environment activated, install the development tools and run the checks with:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m ruff check .
python3 -m black .
python3 -m black --check .
python3 -m pytest
```

The linting and test suite are intentionally lightweight: they cover fast smoke tests, config validation, and stable helper behavior without requiring the real dataset, long training runs, or GPU access.

## Experiment Configuration

The main experiment settings now live in [`configs/experiment.default.json`](configs/experiment.default.json). This file centralizes:

- dataset preparation settings such as `image_size`, `augmentations_per_image`, `samples_per_class`, `test_size`, and `random_state`
- training settings such as `model_names`, `batch_size`, `epochs`, `learning_rate`, `loss`, `validation_size`, `random_state`, and preprocessing selection
- per-model runtime settings such as input channels, effective batch size, dense head size, and dropout
- evaluation settings such as `top_k`
- input/output paths such as the dataset root, artifacts root, and optional report/run overrides

The CLIs load this file by default. You can either edit it directly or point to another JSON file with `--config`.

Example:

```bash
python train.py --config configs/experiment.default.json
python train.py --config configs/experiment.default.json --epochs 3 --models "custom cnn"
```

CLI flags take precedence over the JSON file, so a specific run can be overridden without changing the tracked default configuration.

## Methodological Safeguards

- dataset splitting now happens before augmentation, so synthetic variants of the same source image never leak into the test set
- the splitter prefers patient-level grouping when patient metadata is available, falls back to exam-level grouping when possible, and only uses image-level stratification as a last resort
- training-only augmentations are written under `artifacts/processed/images/train/`, while untouched holdout images are written under `artifacts/processed/images/test/`
- model selection uses a validation subset or train-only cross-validation; the holdout test split is reserved for final evaluation artifacts
- global seeds are fixed for preprocessing and training to reduce run-to-run variance and make results easier to reproduce

## Reproducible Pipeline

### 1. Prepare the dataset

```bash
python preprocess.py
```

Default behavior:

- loads preprocessing and path defaults from `configs/experiment.default.json`
- reads `data/INbreast Release 1.0/INbreast.csv`
- filters to valid BI-RADS labels before generating splits
- infers a grouping strategy from metadata, preferring patient-level split keys when available
- fixes the preprocessing seed with `random_state=42`
- creates the train/test split before augmentation to avoid leakage
- converts DICOMs to normalized `224x224` PNG files
- writes processed images to `artifacts/processed/images/train/` and `artifacts/processed/images/test/`
- creates `3` augmented images per source image only for the training split
- trims only the training split to `35` images per class
- writes splits to:
  - `artifacts/processed/splits/train_split.csv`
  - `artifacts/processed/splits/test_split.csv`
  - `artifacts/processed/splits/split_summary.json`

### 2. Train the models

```bash
python train.py
```

Default behavior:

- loads experiment settings from `configs/experiment.default.json`
- reads the processed split CSVs from `artifacts/processed/splits/`
- maps BI-RADS labels to 8 numeric classes
- runs the configured preprocessing and model registry
- fixes the model seed with `random_state=42`
- uses `validation_size=0.2` when `folds=0`
- uses `folds=4`, `epochs=15`, `batch_size=8`, and `run_skip=True`
- keeps the test split untouched for final predictions; validation happens only inside the training split
- stores artifacts in:
  - `artifacts/runs/history/<preproc_id>/<model_name>/`
  - `artifacts/runs/predictions/<preproc_id>/<model_name>/`

Useful options:

```bash
python train.py --models "custom cnn" --preprocessing none --no-combined-preprocessing
python train.py --folds 0
python train.py --learning-rate 0.0005 --loss categorical_crossentropy
```

### 2b. Run experiments iteratively with safe stop/resume

The iterative runner builds an explicit queue of experiment combinations, persists status after every relevant transition, saves artifacts incrementally after each completed experiment, and resumes without repeating completed work by default.

Command-line usage:

```bash
python run_experiments.py run
python run_experiments.py launch
python run_experiments.py status
python run_experiments.py stop
```

Behavior:

- `run` executes in the foreground
- `launch` starts the same resumable runner in the background
- `stop` requests a graceful stop; the current experiment finishes before the queue pauses
- `status` shows totals, failures, the current task, and where state/results are stored
- completed experiments are skipped automatically on the next run
- failed experiments stay paused unless you pass `--rerun-failed`
- completed experiments are only rerun if you pass `--rerun-completed`

Useful options:

```bash
python run_experiments.py run --rerun-failed
python run_experiments.py run --models "custom cnn" --preprocessing none --no-combined-preprocessing
python run_experiments.py reset
python run_experiments.py reset --purge-results
```

The legacy [`scripts/run_models_loop.sh`](scripts/run_models_loop.sh) file remains only as a compatibility wrapper and now delegates to `run_experiments.py` instead of restarting the process in a fragile loop.

### 2c. Local dashboard with Run and Stop buttons

Start the local dashboard with:

```bash
streamlit run experiment_dashboard.py
```

The dashboard shows:

- total queued combinations
- completed, failed, pending and stopped counts
- the current experiment, when one is running
- the state file, consolidated summary file, history directory, predictions directory and runner log
- Run, Stop, Refresh and Reset controls

The dashboard launches the same `run_experiments.py` runner in the background, so CLI and UI stay consistent over the same persisted state.

### 3. Aggregate the report

```bash
python evaluate.py
```

Default behavior:

- loads evaluation defaults from `configs/experiment.default.json`
- reads run outputs from `artifacts/runs/`
- computes holdout metrics such as accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC, top-k accuracy, and derived rankings
- saves per-run artifacts including classification reports and confusion matrices under `artifacts/reports/evaluation_details/`
- writes the final report to:
  - `artifacts/reports/final_comprehensive_results.csv`

## Notebooks

The notebooks are still useful, but their role is now lighter:

- [`notebooks/data.ipynb`](notebooks/data.ipynb): dataset exploration and augmentation preview using the shared config loader
- [`notebooks/run-models.ipynb`](notebooks/run-models.ipynb): thin training demo using `pipeline.train` and the shared experiment config
- [`notebooks/post-trainning-analysis.ipynb`](notebooks/post-trainning-analysis.ipynb): report generation and result inspection from the same config source

## Legacy Compatibility

The old modules under `utils/` remain as compatibility wrappers that forward to the new `pipeline/` implementation. The new package is the source of truth.

## Stored Artifacts

Generated artifacts now live under `artifacts/`:

- `artifacts/processed/images/`
- `artifacts/processed/splits/`
- `artifacts/runs/history/`
- `artifacts/runs/predictions/`
- `artifacts/experiments/state/`
- `artifacts/experiments/logs/`
- `artifacts/experiments/summary/`
- `artifacts/experiments/tasks/`
- `artifacts/reports/evaluation_details/`
- `artifacts/reports/final_comprehensive_results.csv`

The iterative runner additionally persists:

- queue and task state in `artifacts/experiments/state/runner_state.json`
- graceful stop control in `artifacts/experiments/control/stop_requested.flag`
- a consolidated per-task summary in `artifacts/experiments/summary/experiment_runs.csv`
- per-task metadata snapshots in `artifacts/experiments/tasks/`
- runner logs in `artifacts/experiments/logs/iterative-runner.log`

The tracked file [`data/final_comprehensive_results.csv`](data/final_comprehensive_results.csv) is kept only as a historical artifact from the previous workflow.

## Research-Use Note

This repository is intended for academic and research use only. It is not a clinical device, not a validated diagnostic tool, and must not be used to make medical decisions.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
