# Prediction Layer

`next_allocation.py` trains the configured Model B policy on the available
history and produces two signals: a myopic latest-return reference and the
published Model B strategic allocation.

The published allocation is the deterministic modal target across all 286
possible starting portfolios for the current market state. Its `score` is the
modal share, not an expected return. The current user portfolio remains useful
for execution diagnostics, but does not change the published strategic target.

## Runner and configuration

Run the product entry point from repository root:

```bash
uv run python src/prediction/run_prediction.py
```

The only command-line option is:

```bash
--feature-mode {compact_regime,raw,normalized_binned,tercile_binned}
```

`compact_regime` is the default and the supported Model B product state. The
other modes remain research compatibility options and do not change the Model B
product contract.

Training parameters are intentionally not duplicated as CLI arguments. The
runner reads them from:

```text
analysis/hyperparameter_search/current/best_hyperparameters.yaml
```

That file supplies `episodes`, `training_mode` (`historical` or `synthetic`),
seed, learning rate and schedule, discount factor, epsilon schedule, and state
precision. The current product configuration is the selected Model B setup from
trial 169: 54,000 synthetic episodes, seed 42, constant learning rate 0.05,
discount factor 0.95, epsilon 0.50 decaying to 0.02, and `compact_regime`.
Its initial HPO ranking predated the corrected cost-aware evaluator; the
committed selected-model run records the corrected-policy evaluation and the
fixed configuration remains fully reproducible.

The runner currently uses an equal-weight execution reference
`[0.25, 0.25, 0.25, 0.25]`. This reference is retained for myopic diagnostics;
the published Model B modal target itself is computed across all 286 origin
portfolios and therefore does not depend on that assumption.

The runner writes dated artifacts under `analysis/forecasts/standard/`:

- `allocation_prediction.csv`: the published Model B modal target.
- `allocation_predictions.csv`: modal target plus the myopic reference.
- `trained_model.json`: Q-table used for the forecast.
- `training_history.csv` and `prediction_metadata.json`: reproducibility data.
