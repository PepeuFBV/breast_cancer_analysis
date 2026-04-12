# Breast Cancer Analysis

This repository contains a notebook-first research workflow for mammogram classification with deep learning. The project studies how image preprocessing affects multi-class BI-RADS classification on mammography images and compares a custom CNN against transfer learning architectures such as ResNet, DenseNet, EfficientNet, MobileNetV3, Inception, NASNet, CheXNet, and VGG19.

The repository also includes the paper source in [`article/main.tex`](article/main.tex) and the compiled PDF in [`article/main.pdf`](article/main.pdf).

## Project objective

The main goal of the project is to evaluate whether simple, reproducible preprocessing operations improve classification performance on mammography images. The current workflow:

- converts INbreast DICOM images to normalized PNG files,
- applies data augmentation and a balanced sampling strategy,
- trains multiple model architectures on different preprocessing variants,
- stores per-run histories and predictions, and
- aggregates results into a final CSV for comparison.

## Dataset actually used

The code in this repository is built around **INbreast Release 1.0**

Important details inferred from the notebooks:

- `data.ipynb` reads metadata from `data/INbreast Release 1.0/INbreast.csv`.
- Raw images are read from `data/INbreast Release 1.0/AllDICOMs/*.dcm`.
- Valid labels are the BI-RADS classes `1`, `2`, `3`, `4a`, `4b`, `4c`, `5`, and `6`.
- `run-models.ipynb` maps those labels to 8 numeric classes before training.

The dataset is not versioned in Git. You must obtain it separately and place it in the exact directory structure above before running the full pipeline.

## Prerequisites

Before running the project, make sure you have:

- Python 3 available locally
- `pip`
- enough disk space for the INbreast dataset plus generated PNG files and results
- a Jupyter environment if you want to execute the notebooks interactively

Practical notes:

- The training workflow depends on TensorFlow/Keras and is much more realistic to run on a machine with GPU support.
- Dependency versions in `requirements.txt` are not pinned, so environment drift is possible.
- Notebook metadata in the repository mixes Python 3.11 and 3.12, so exact environment parity is not guaranteed.
- `requirements.txt` includes notebook-related Python packages such as `ipykernel`, `nbconvert`, and `papermill`, but does not pin a full Jupyter Notebook/Lab frontend setup.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/PepeuFBV/breast_cancer_analysis.git
cd breast_cancer_analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then place the INbreast dataset under:

```text
data/INbreast Release 1.0/
```

with both:

- `INbreast.csv`
- `AllDICOMs/*.dcm`

If you plan to execute the notebooks interactively, use an existing Jupyter Notebook or JupyterLab installation in the same virtual environment.

## Execution flow

This project is organized around notebooks. The intended order is:

### 1. Prepare the dataset

Run [`notebooks/data.ipynb`](notebooks/data.ipynb).

What this notebook does:

- reads `INbreast.csv`,
- loads DICOM files from `AllDICOMs/`,
- normalizes images and resizes them to `224x224`,
- exports original and augmented PNG files to `data/augmented_images/`,
- trims each class to `samples_per_class = 35`,
- creates a stratified train/test split, and
- writes `data/train_split.csv` and `data/test_split.csv`.

This step must be completed before model training.

### 2. Train the models

Run [`notebooks/run-models.ipynb`](notebooks/run-models.ipynb).

The notebook:

- loads `data/train_split.csv` and `data/test_split.csv`,
- maps BI-RADS labels to 8 classes,
- applies preprocessing functions from [`utils/preprocessing.py`](utils/preprocessing.py),
- builds models defined in [`utils/models.py`](utils/models.py),
- trains each model for `15` epochs, and
- saves histories and predictions under `data/results/`.

Default execution behavior to know about:

- `fold = 4` by default, so the notebook merges `train_split.csv` and `test_split.csv` and then performs `StratifiedKFold` cross-validation on the merged dataset.
- If you want to evaluate the fixed train/test split directly, you must change `fold` to `0` inside the notebook before running it.
- `run_skip = True` is enabled by default, so runs with both history and prediction CSVs already present are skipped.

For long unattended runs, the repository provides a helper script:

```bash
bash scripts/run_models_loop.sh
```

This script executes `run-models.ipynb` with Papermill and writes a log in `notebooks/log.txt`.

### 3. Aggregate and inspect results

Run [`notebooks/post-trainning-analysis.ipynb`](notebooks/post-trainning-analysis.ipynb).

This notebook expects the outputs generated in step 2 and:

- loads `data/results/history/`,
- joins per-run metrics,
- reads prediction files from `data/results/predictions/`,
- computes additional ranking and top-k statistics, and
- writes `data/final_comprehensive_results.csv`.

## Where results are stored

The workflow writes artifacts to the following locations:

- `data/augmented_images/`: PNG images generated from the DICOM dataset
- `data/train_split.csv` and `data/test_split.csv`: split definitions created by `data.ipynb`
- `data/results/history/<preproc_id>/<model_name>/`: best-epoch history CSVs per experiment
- `data/results/predictions/<preproc_id>/<model_name>/`: prediction CSVs per experiment
- `data/final_comprehensive_results.csv`: aggregated summary produced by `post-trainning-analysis.ipynb`

The repository currently tracks `data/final_comprehensive_results.csv`, but it does **not** track the intermediate `data/results/` directory used to build it.

## Research-use note

This repository is intended for **academic and research use only**. It is **not** a clinical device, not a validated diagnostic tool, and must not be used to make medical decisions.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
