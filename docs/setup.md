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

Recommended unattended bootstrap:

```bash
python3 scripts/bootstrap_env.py --gpu auto
```

Use `--gpu required` to fail if TensorFlow cannot be configured for GPU, and
add `--dev` to install development dependencies in the same pass.

After bootstrapping, either activate the environment or use `./.venv/bin/python`
directly.

Manual fallback:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

For tests and formatting:

```bash
python3 scripts/bootstrap_env.py --gpu auto --dev
```

The full `run_experiments.py run` battery can take hours. For default setup
validation, run smoke and long-run checks first:

```bash
./.venv/bin/python scripts/check_runtime.py --device auto
./.venv/bin/python scripts/validate_long_runner.py --combinations 20 --device cpu
```

CPU-only validation is supported. GPU is optional unless you explicitly pass a
`--require-gpu` flag to the runtime checks.

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
./.venv/bin/python scripts/check_environment.py --require-venv
```

If the dataset is not present yet:

```bash
./.venv/bin/python scripts/check_environment.py --require-venv --skip-dataset
```
