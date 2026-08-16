# Matched-Context Policy Responses

This is a finite-state partial-dependence-style analysis, not SHAP and not a causal estimate. For each comparison, the other three market variables are held fixed across every compatible discrete context; the reported change is the average change in selected target weight.

## Low-to-High Matched Responses

| Feature | Asset | Average target-weight change | Contexts with same sign |
| --- | --- | ---: | ---: |
| momentum_strength | w_arg | -3.1 pp | 39% |
| momentum_strength | w_ced | +12.5 pp | 61% |
| momentum_strength | w_sp500 | -3.6 pp | 53% |
| momentum_strength | w_gold | -5.8 pp | 47% |
| usd_ars_regime | w_arg | +0.0 pp | 39% |
| usd_ars_regime | w_ced | +2.5 pp | 47% |
| usd_ars_regime | w_sp500 | +2.8 pp | 50% |
| usd_ars_regime | w_gold | -5.3 pp | 42% |
| vix_regime | w_arg | +5.0 pp | 56% |
| vix_regime | w_ced | -3.1 pp | 44% |
| vix_regime | w_sp500 | +1.4 pp | 39% |
| vix_regime | w_gold | -3.3 pp | 47% |

## Reading the Tables

- `policy_feature_level_means.csv` shows the average target allocation at each feature level.
- `policy_feature_matched_effects.csv` gives every pairwise comparison, the number of fixed contexts, average/median percentage-point change, and the share of contexts where the direction is positive, negative, or unchanged.
- Momentum leader is categorical, so its comparisons are leader-to-leader changes rather than a low-to-high gradient.
