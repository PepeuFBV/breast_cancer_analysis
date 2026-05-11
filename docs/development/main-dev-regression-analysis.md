# Main vs Dev Regression Analysis

This note summarizes why the `dev` branch runner behavior differs from `main` and which stabilization work was added. It is historical context for maintainers.

## Purpose

Capture branch-level regression reasoning around long-run lifecycle and memory behavior.

## Summary

- `main` used notebook-driven execution with process restarts on failure.
- `dev` moved to reusable Python modules and a long-lived iterative runner (`run_experiments.py` + `pipeline/experiments/runner.py`).
- Failures after several combinations were analyzed as lifecycle/resource-retention risks in long-lived processes rather than queue-shape bugs.
- `dev` added stronger cleanup paths, runtime checks, and long-run regression validation coverage.

## Related docs

- Canonical docs map: [../index.md](../index.md)
- Runner command reference: [../execution.md](../execution.md)
- Testing and long-run validation: [../testing.md](../testing.md)
