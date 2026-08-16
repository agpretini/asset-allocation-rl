# State and Cost Formulation Diagnostic

`run.py` compares three deliberately narrow tabular Q-learning formulations.
It is a diagnostic experiment: it does not modify the production model, its
reward implementation, the selected trial, or any saved Q-table. The completed
analysis establishes Model B as the principal formulation for the next product
iteration, subject to the limits documented below.

## Decision record

**Decision:** adopt **Model B — market-only state with rebalance-cost-aware
learning and action selection** as the principal model architecture. Keep C as
a frictionless strategic control and retain A only as the historical baseline.

The decision is based on the completed synthetic-training diagnostic using the
selected 18,000-episode configuration, seed 42, the compact-regime state, and
the 2023-12 through 2025-11 historical holdout. All figures below are net of
realised 1.4% turnover costs.

| Model | Cumulative return | CAGR | Sharpe | Maximum drawdown | Average turnover | Total rebalance cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A — baseline | 51.3% | 24.1% | 1.90 | -7.9% | 39.1% | 12.6% |
| B — selected | **91.9%** | **40.5%** | 2.10 | -4.5% | **22.0%** | **7.1%** |
| C — no-cost control | 57.4% | 26.7% | **2.19** | **-4.3%** | 45.4% | 14.6% |

B has the highest cumulative return and CAGR, halves A's turnover and trading
cost, and materially improves drawdown. Its Sharpe is higher than A's, though
not higher than C's or the strongest static/rule-based baselines in this single
window. Coverage provides the central structural justification: B visited
88.1% of its possible state-action pairs, versus 3.7% for A. This substantially
reduces the sparse-Q-table behavior that made A's recommendation vary wildly
with the initial allocation.

This is evidence for a product choice, not a final unbiased investment claim:
the selected configuration was informed by prior test-based HPO and the
comparison uses one 23-month historical holdout. Any release decision should
be rechecked on fresh walk-forward windows.

## Product policy rule

Model B learns only the market state. At inference, for a market state `m` and
a starting portfolio `p`, it chooses:

```text
target(m, p) = argmax_a [Q(m, a) - 0.014 × turnover(p, a)]
```

The product-facing target must be stable even when the user's exact starting
portfolio is not one of the 286 discrete allocations. Therefore the published
strategic target for each market state is the **mode** of the B targets across
all 286 possible starting portfolios:

```text
product_target(m) = mode({ target(m, p) : p ∈ 286 action-grid portfolios })
```

If two targets have the same modal count, select the lower action ID, matching
the project's deterministic tie-breaking convention. This is a strategic
allocation rule; it intentionally summarizes away user-specific execution
costs. The exact B decision rule remains available for diagnostics and future
execution-layer work.

Across all 108 compact market scenarios, the median modal share is **74.5%**
(reported operationally as **74%**) of the 286 starting portfolios. In other
words, in the median scenario, roughly three out of four possible starting
portfolios select the same cost-aware target. The range is 10.1% to 100.0%; a
low share is an explicit signal that a regime has material starting-portfolio
sensitivity and that its modal target should be interpreted cautiously. See
`B_modal_target_share_histogram.svg` and
`full_market_policy_B_market_state_cost_at_decision.csv`.

## Models

- **A — baseline:** compact market state plus current portfolio; the actual
  production CRRA utility (gamma = 0.5) of net return including the rebalance
  cost; greedy `argmax Q`.
- **B — market-only + execution cost:** compact market state only; the same
  CRRA utility of net return including the rebalance cost; at inference choose
  `argmax(Q - 0.014 × turnover(current, target))`.
- **C — market-only control:** compact market state only; CRRA utility of
  gross return; greedy `argmax Q`, with no cost used by the policy.

The realised historical backtest charges the same 1.4% turnover cost to all
three policies. This reports executable net performance and makes C a control
for decision-time cost awareness rather than a free-trading strategy.

All three models use the same CRRA curvature (gamma = 0.5). A versus B compares
the full proposed architecture change: removal of the current portfolio from
the learned state plus B's explicit decision-time cost adjustment. B versus C
isolates the complete effect of cost treatment because they share the reduced
state: B accounts for costs in both learning and decision making, whereas C
ignores them in both places.

## Baseline reuse

The legacy Model A trial-371 checkpoint saved under `../selected_synthetic_trial_371`
is a full-data model. The runner reuses it for the fixed latest-market
sensitivity table for A, avoiding unnecessary retraining there. It cannot be
used for the configured historical holdout: it contains later observations and
the old split-level models/action paths were not retained. Therefore the runner
re-trains A, B, and C under the same train window to produce complete and fair
historical metrics.

For the original Model A current-market coverage investigation, run:

```bash
uv run python analysis/diagnostics/model_formulations/diagnose_legacy_trial_371.py
```

It writes regenerable output under `results/`; it is not part of the selected
Model B production run.

## Run

From repository root:

```bash
uv run python analysis/diagnostics/model_formulations/run.py
```

Defaults replicate selected walk-forward split 1's final policy window:
train from 2018-12 through 2023-11 and evaluate 2023-12 through 2025-11.
All use 18,000 synthetic episodes, seed 42, the selected learning parameters,
and the existing market simulator. The run trains five agents: A/B/C for the
historical comparison and B/C on the full data for the current-market panel.
It may therefore take substantially longer than a single 18,000-episode run.

Override only the shared protocol inputs when needed:

```bash
uv run python analysis/diagnostics/model_formulations/run.py \
  --episodes 2000 --seed 42 \
  --train-start 2018-12-31 --train-end 2023-11-30 \
  --evaluation-start 2023-12-31 --evaluation-end 2025-11-30 \
  --output-dir analysis/diagnostics/model_formulations/results_smoke
```

## Outputs

- `historical_metrics.csv`: cumulative return, CAGR, annualized volatility,
  Sharpe, Sortino, maximum drawdown, turnover, total rebalance cost, portfolio
  changes, and distinct selected targets.
- `coverage.csv`: theoretical and visited state/state-action counts and
  coverage percentages.
- `visit_distributions.csv`: p10/p25/p50/p75/p90 and median visits for states
  and state-action pairs.
- `current_market_sensitivity.csv`: all 286 starting portfolios under the
  latest observed market state, with target, Q-value, decision score, turnover,
  and cost for each model.
- `summary.md` and `configuration.json`: an auditable, concise interpretation
  and exact run settings.
- `models/`: three historical-evaluation checkpoints plus B/C full-data
  checkpoints for the latest-market panel. They are outputs, not selected
  production models.

## Policy charts

After diagnostic training has completed, render the August 2026 policy for all
286 discrete starting allocations plus the actual 25/25/25/25 allocation:

```bash
uv run python analysis/diagnostics/model_formulations/render_current_market_policy.py
```

It writes `current_market_policy_B_market_state_cost_at_decision.svg` and
`current_market_policy_C_market_state_no_cost.svg`, with adjacent CSV files.
For B, the target uses the explicit rebalance-cost adjustment; for C, it is the
raw greedy Q target. Because B/C learn a market-only state, the training-visit
count is shared across every starting allocation in each chart.

## Historical comparison against baselines

Render the completed diagnostic holdout comparison without retraining:

```bash
uv run python analysis/diagnostics/model_formulations/render_historical_comparison.py
```

The output contains `historical_model_vs_baselines_cumulative_returns.svg` and
`historical_model_vs_baselines_metrics.csv`. The latter includes cumulative
return, CAGR, annualized volatility, Sharpe, Sortino, drawdown, and turnover.

## Full market-policy maps

Render all 108 compact market scenarios without re-training:

```bash
uv run python analysis/diagnostics/model_formulations/render_full_market_policy.py
```

`full_market_policy_B_market_state_cost_at_decision.svg` shows the modal
cost-aware target across the 286 possible starting allocations for each market
regime. `full_market_policy_C_market_state_no_cost.svg` shows C's unique greedy
target for each market regime. The companion CSVs include the modal share and
the number of distinct B targets behind each visual cell.

To inspect how representative B's modal target is across the 108 regimes:

```bash
uv run python analysis/diagnostics/model_formulations/render_b_modal_share_histogram.py
```

This writes `B_modal_target_share_histogram.svg`. A high share means most
starting portfolios agree on the same B target for that market regime. A PNG
preview can be generated locally for editors that open SVG files as text.
