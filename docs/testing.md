# Testing

Default pytest runs fast deterministic tests and excludes `slow` and
`integration` markers.

```bash
python -m pytest
```

Useful targeted checks:

```bash
python -m pytest -m unit
python -m pytest -m smoke
python -m pytest -m gpu
python -m pytest tests/test_iterative_runner.py
```

Formatting and lint:

```bash
python -m ruff check .
python -m black --check .
```

Project smoke checks:

```bash
python scripts/check_environment.py --require-venv
python scripts/validate_dataset.py
python scripts/check_gpu.py
python scripts/smoke_run.py
```

`scripts/smoke_run.py` uses synthetic data and a tiny mocked model path. It
checks config resolution, runner state, training artifact writing, and
evaluation report generation without the full INbreast experiment grid.

After `python preprocess.py`, you can run one real training task:

```bash
python run_experiments.py run \
  --models "custom cnn" \
  --preprocessing none \
  --no-combined-preprocessing \
  --folds 0 \
  --epochs 1 \
  --limit 1
```

Do not use the full default experiment queue as the default validation path.
