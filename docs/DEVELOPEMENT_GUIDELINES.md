# Portfolio MDP - Development Guidelines

## Purpose

This document defines coding standards, tooling decisions, development practices, and engineering principles for the project.

All contributors and AI coding agents should follow these guidelines.

---

# Core Philosophy

The project prioritizes:

1. Correctness.
2. Simplicity.
3. Readability.
4. Reproducibility.
5. Performance.

When tradeoffs exist:

```text
Correctness > Readability > Performance > Cleverness
```

Avoid clever code that is difficult to understand.

Prefer simple and explicit implementations unless performance becomes a proven bottleneck.

---

# Python Version

The project should target:

```text
Python >= 3.12
```

---

# Dependency Management

The project uses:

```text
uv
```

for dependency management and virtual environments.

Do not use:

- pipenv
- poetry
- conda

unless explicitly approved.

## Environment Setup

Create environment:

```bash
uv venv
```

Activate environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

Add dependency:

```bash
uv add pandas
```

Add development dependency:

```bash
uv add --dev pytest
```

Update lockfile:

```bash
uv lock
```

---

# Package Management

Dependencies should be declared in:

```text
pyproject.toml
```

Avoid:

```text
requirements.txt
```

as the primary dependency source.

A requirements file may be generated for compatibility purposes if needed.

---

# Testing Strategy

The project uses:

```text
pytest
```

for testing.

## Test Coverage

All core logic must have tests.

Required coverage areas:

- Action space generation.
- Reward calculation.
- Portfolio environment.
- Evaluation metrics.
- Benchmark calculations.
- Feature engineering.

## Test Structure

```text
tests/
├── test_action_space.py
├── test_reward.py
├── test_environment.py
├── test_metrics.py
├── test_benchmarks.py
└── test_features.py
```

## Running Tests

Run all tests:

```bash
pytest
```

Run a single file:

```bash
pytest tests/test_reward.py
```

## Testing Principle

Business logic should be tested independently.

Avoid tests that require training a complete RL agent.

Prefer:

```python
assert calculate_reward(...) == expected_value
```

over large integration tests whenever possible.

---

# Code Formatting

The project uses:

```text
ruff
```

for linting and formatting.

## Installation

```bash
uv add --dev ruff
```

## Format Code

```bash
ruff format .
```

## Lint Code

```bash
ruff check .
```

## Fix Lint Issues

```bash
ruff check . --fix
```

---

# Type Hints

All public functions should include type hints.

Example:

```python
def calculate_reward(
    portfolio_return: float,
    turnover: float,
    transaction_cost: float,
) -> float:
    ...
```

Avoid untyped public APIs.

---

# Project Structure

Code should live inside:

```text
src/
```

Avoid placing production logic inside notebooks.

Preferred structure:

```text
src/
├── agents/
├── backtesting/
├── config/
├── data/
├── env/
├── evaluation/
├── features/
└── utils/
```

---

# Notebook Usage

Notebooks are allowed only for:

- Exploration.
- Prototyping.
- Visualization.

Notebooks should never become the source of truth.

Once logic stabilizes, move it into:

```text
src/
```

and import it from notebooks.

---

# Performance Guidelines

The project should prioritize fast iteration cycles.

## Preferred Libraries

Numerical operations:

```python
numpy
```

Tabular operations:

```python
pandas
```

Machine learning:

```python
scikit-learn
```

Deep learning:

```python
torch
```

Visualization:

```python
matplotlib
```

Avoid introducing additional frameworks without clear justification.

---

# Performance Principles

Prefer:

- Vectorized operations.
- Batch calculations.
- Precomputed features.

Avoid:

- Deep nested loops.
- Repeated dataframe copies.
- Unnecessary object creation.

Bad:

```python
for row in df.iterrows():
    ...
```

Preferred:

```python
df["signal"] = ...
```

using vectorized operations.

---

# Configuration Management

Configuration should not be hardcoded.

Use:

```text
yaml
```

files stored under:

```text
src/config/
```

Examples:

```text
assets.yaml
training.yaml
backtesting.yaml
```

---

# Reproducibility

All experiments must be reproducible.

Every training script should:

- Accept a random seed.
- Save configuration used.
- Save model artifacts.
- Save evaluation metrics.

Example:

```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```

---

# Logging

Avoid excessive print statements.

Use:

```python
import logging
```

Example:

```python
logger.info("Training started")
logger.warning("Missing feature detected")
```

---

# Experiment Tracking

Each experiment should create a dedicated folder.

Example:

```text
experiments/
└── experiment_001/
    ├── config.yaml
    ├── metrics.json
    ├── portfolio_values.csv
    ├── actions.csv
    └── notes.md
```

The goal is to make every result reproducible.

---

# Documentation

All modules should contain:

- Module-level docstrings.
- Function docstrings.
- Type hints.
- README explaining the module logic and its way of working.

Example:

```python
def calculate_turnover(
    current_weights: np.ndarray,
    previous_weights: np.ndarray,
) -> float:
    """
    Compute portfolio turnover between two allocations.

    Parameters
    ----------
    current_weights : np.ndarray
    previous_weights : np.ndarray

    Returns
    -------
    float
        Portfolio turnover.
    """
```

---

# AI Agent Guidelines

AI coding agents should:

1. Read `PRODUCT_SPEC.md` before implementing features.
2. Read `ARCHITECTURE.md` before creating modules.
3. Follow this document for tooling and coding standards.
4. Prefer modifying existing modules over creating duplicates.
5. Write tests alongside implementation.
6. Avoid introducing unnecessary abstractions.
7. Keep implementations small and inspectable.

---

# Dependencies

Initial recommended dependencies:

```text
numpy
pandas
scikit-learn
torch
pyyaml
matplotlib
pytest
ruff
```

Additional dependencies should only be added when they provide clear value.

---

# Definition of Done

A feature is considered complete only if:

- Implementation exists.
- Tests pass.
- Type hints are present.
- Linting passes.
- Documentation is updated.
- No data leakage is introduced.
- Results are reproducible.

Code that works but is not tested is not considered finished.