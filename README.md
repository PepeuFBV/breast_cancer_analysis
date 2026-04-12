# Breast Cancer Analysis

This repository contains a reproducible Python pipeline for mammogram classification experiments based on the INbreast dataset. The preprocessing, training, and evaluation logic now lives in reusable Python modules under `pipeline/`, while the notebooks remain available as exploratory and reporting interfaces.

The repository also includes the paper source in [`article/main.tex`](article/main.tex) and the compiled PDF in [`article/main.pdf`](article/main.pdf).

## What Changed

The project used to be notebook-first. The main workflow is now organized around Python entrypoints:

- `preprocess.py` builds processed images and train/test splits
- `train.py` runs the experiment grid and stores model artifacts
- `evaluate.py` aggregates run outputs into a final report

The notebooks in [`notebooks/`](notebooks) now consume those modules instead of owning the full pipeline logic.

## Project Layout

```text
pipeline/
  data/
  train/
  evaluate/
  utils/
artifacts/
  processed/
  runs/
  reports/
data/
  INbreast Release 1.0/
notebooks/
preprocess.py
train.py
evaluate.py
```

## Dataset

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

## Reproducible Pipeline

### 1. Prepare the dataset

```bash
python preprocess.py
```

Default behavior:

- reads `data/INbreast Release 1.0/INbreast.csv`
- filters to valid BI-RADS labels before generating splits
- converts DICOMs to normalized `224x224` PNG files
- writes processed images to `artifacts/processed/images/`
- creates `3` augmented images per source image
- trims each class to `35` samples
- writes splits to:
  - `artifacts/processed/splits/train_split.csv`
  - `artifacts/processed/splits/test_split.csv`

### 2. Train the models

```bash
python train.py
```

Default behavior:

- reads the processed split CSVs from `artifacts/processed/splits/`
- maps BI-RADS labels to 8 numeric classes
- runs the configured preprocessing and model registry
- uses `folds=4`, `epochs=15`, `batch_size=8`, and `run_skip=True`
- stores artifacts in:
  - `artifacts/runs/history/<preproc_id>/<model_name>/`
  - `artifacts/runs/predictions/<preproc_id>/<model_name>/`

Useful options:

```bash
python train.py --models "custom cnn" --preprocessing none --no-combined-preprocessing
python train.py --folds 0
```

For long unattended runs, use:

```bash
bash scripts/run_models_loop.sh
```

### 3. Aggregate the report

```bash
python evaluate.py
```

Default behavior:

- reads run outputs from `artifacts/runs/`
- computes top-k metrics and derived rankings
- writes the final report to:
  - `artifacts/reports/final_comprehensive_results.csv`

## Notebooks

The notebooks are still useful, but their role is now lighter:

- [`notebooks/data.ipynb`](notebooks/data.ipynb): dataset exploration and augmentation preview
- [`notebooks/run-models.ipynb`](notebooks/run-models.ipynb): thin training demo using `pipeline.train`
- [`notebooks/post-trainning-analysis.ipynb`](notebooks/post-trainning-analysis.ipynb): report generation and result inspection

## Legacy Compatibility

The old modules under `utils/` remain as compatibility wrappers that forward to the new `pipeline/` implementation. The new package is the source of truth.

## Stored Artifacts

Generated artifacts now live under `artifacts/`:

- `artifacts/processed/images/`
- `artifacts/processed/splits/`
- `artifacts/runs/history/`
- `artifacts/runs/predictions/`
- `artifacts/reports/final_comprehensive_results.csv`

The tracked file [`data/final_comprehensive_results.csv`](data/final_comprehensive_results.csv) is kept only as a historical artifact from the previous workflow.

## Research-Use Note

This repository is intended for academic and research use only. It is not a clinical device, not a validated diagnostic tool, and must not be used to make medical decisions.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
