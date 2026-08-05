# Simulation Layer

This layer generates synthetic monthly training episodes without replacing the
historical evaluation pipeline.

The model is fitted exclusively on the current walk-forward training window:

1. Fixed VIX terciles define `LOW_VOL`, `MEDIUM_VOL`, and `HIGH_VOL`.
2. A smoothed monthly Markov transition matrix models regime persistence.
3. Each regime receives a separate regularized multivariate Student-t model.

The Student-t vector contains only primitive variables: four asset log returns,
monthly log change in USD/ARS, and monthly log change in VIX. Synthetic prices,
USD/ARS, and VIX are reconstructed from those values. The existing
`StateBuilder` then calculates the final features and portfolio state.

`MarketSimulator.stream_episodes()` is lazy and reproducible from the configured
seed. Synthetic observations are used only for Q-learning updates. Validation
and test paths remain real historical data.

Configuration lives in `src/config/simulation.yaml`. The default Student-t
degrees of freedom is 8, and covariance eigenvalues are floored by the configured
regularization value.
