# Contributing to HistoMoE

Thank you for your interest in contributing! HistoMoE is a GSoC 2025 project
in active development and welcomes contributions from the community.

## Getting Started

1. **Fork** the repository and clone your fork locally.
2. Create a **feature branch**: `git checkout -b feat/your-feature-name`
3. **Install dependencies** in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Coding Standards

- **Style**: We use [Black](https://black.readthedocs.io/) (line length 100) and
  [Ruff](https://docs.astral.sh/ruff/) for linting.
  ```bash
  black histomoe/
  ruff check histomoe/ --fix
  ```
- **Type hints**: All public functions should use Python 3.9+ type annotations.
- **Docstrings**: We follow NumPy docstring format. Every public function, class,
  and method needs a docstring with Parameters and Returns sections.

## Adding Tests

All new features must include unit tests in the `tests/` directory using `pytest`.
Use the shared fixtures from `tests/conftest.py` to avoid real data dependencies.

```bash
# Run tests before opening a PR
pytest tests/ -v --tb=short
```

## Pull Request Process

1. Ensure all tests pass: `pytest tests/ -v`
2. Ensure code is formatted: `black --check histomoe/`
3. Write a clear PR description explaining **what** and **why**.
4. Link any related issues.

## Project Areas

| Area | Files | Good first issues |
|------|-------|-------------------|
| New backbone support | `vision_encoder.py` | Add pathology ViT (UNI, CONCH) |
| Gating strategies | `gating_network.py` | Implement hard-switch MoE |
| New metrics | `training/metrics.py` | Add SSIM for spatial maps |
| Datasets | `data/st_dataset.py` | Support Slide-seq format |
| Visualization | `visualization/` | Interactive Plotly routing viz |

## Questions?

Open a [GitHub Discussion](https://github.com/your-org/histomoe/discussions) or
file an issue — we're happy to help!
