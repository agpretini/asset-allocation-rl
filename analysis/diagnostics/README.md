# Diagnostics

This folder contains focused, reproducible investigations of model behavior.
Unlike `analysis/runs/`, a diagnostic is not a candidate production run and its
outputs must not be treated as a selected investment policy.

- `model_formulations/`: isolates the effect of including the current
  allocation in the tabular state and of applying transaction costs during
  learning versus action selection.
