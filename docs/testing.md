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
python -m pytest -m memory
python -m pytest -m stress
python -m pytest -m "not slow and not integration and not gpu"
python -m pytest tests/test_iterative_runner.py
python -m pytest tests/test_long_execution_regression.py -v
python -m pytest tests/test_memory_cleanup.py -v
```

Marker intent:

- `unit`: fast deterministic coverage
- `smoke`: quick end-to-end synthetic validation
- `memory`: cleanup and bounded-growth checks
- `stress`: longer mocked queue/resume coverage
- `gpu`: mocked GPU/runtime behavior checks

Formatting and lint:

```bash
python -m ruff check .
python -m black --check .
```

Project smoke checks:

```bash
./.venv/bin/python scripts/check_environment.py --require-venv
./.venv/bin/python scripts/validate_dataset.py
./.venv/bin/python scripts/check_gpu.py
./.venv/bin/python scripts/check_runtime.py --device cpu
./.venv/bin/python scripts/check_runtime.py --device auto
./.venv/bin/python scripts/check_runtime.py --device gpu || true
./.venv/bin/python scripts/smoke_run.py
./.venv/bin/python scripts/validate_long_runner.py --combinations 20 --device cpu
./.venv/bin/python scripts/validate_long_runner.py --combinations 20 --device auto
```

`scripts/smoke_run.py` uses synthetic data and a tiny mocked model path. It
checks config resolution, runner state, training artifact writing, and
evaluation report generation without the full INbreast experiment grid.

`scripts/validate_long_runner.py` uses tiny real training tasks and should be
the default validation path for long-run stability beyond 8 or 9 combinations.

After `python preprocess.py`, you can run one real training task:

```bash
./.venv/bin/python run_experiments.py run \
  --models "custom cnn" \
  --preprocessing none \
  --no-combined-preprocessing \
  --folds 0 \
  --epochs 1 \
  --limit 1
```

Do not use the full default experiment queue as the default validation path.
It can take hours. Use smoke/memory checks first and only run the full queue
after validation passes.
