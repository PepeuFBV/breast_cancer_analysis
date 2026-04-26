# Setup

Use Python 3.10, 3.11, or 3.12. The project metadata intentionally excludes
Python 3.13 until the TensorFlow wheel stack is validated there.

## OS Packages

Debian/Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip build-essential python3-dev
```

On other Linux distributions, install the equivalent Python, venv, pip, compiler,
and Python header packages.

## Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For tests and formatting:

```bash
python -m pip install -r requirements-dev.txt
```

## Dataset

Place INbreast here:

```text
data/INbreast Release 1.0/
  INbreast.csv
  AllDICOMs/*.dcm
```

Validate it:

```bash
python scripts/validate_dataset.py
```

## Setup Check

Run this before preprocessing or training:

```bash
python scripts/check_environment.py --require-venv
```

If the dataset is not present yet:

```bash
python scripts/check_environment.py --require-venv --skip-dataset
```
