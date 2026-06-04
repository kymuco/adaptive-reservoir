# Development

This document describes the basic local development workflow for `adaptive-reservoir`.

## Install locally

```bash
python -m pip install -e ".[dev]"
```

## Run checks

Run the same checks used by CI:

```bash
python -m ruff check .
python -m pytest
```

## Optional pre-commit workflow

The project does not require a pre-commit framework yet. For now, use the same commands before pushing:

```bash
python -m ruff check .
python -m pytest
```

If a pre-commit configuration is added later, it should remain a thin wrapper around the same checks instead of introducing a separate local-only quality gate.

## Branch policy after M0

M0 was allowed to land directly on `main` while the repository was being created.

Starting from M1, changes should be developed on short-lived branches and reviewed before merging into `main`.
