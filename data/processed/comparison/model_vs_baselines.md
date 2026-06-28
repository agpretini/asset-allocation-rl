# Model Vs Baselines

## Metric Table
| index | kind | cagr | volatility | sharpe | sortino | maximum_drawdown | average_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_learning | model | 0.1838 | 0.2070 | 0.9234 | 1.4241 | -0.2894 | 0.5466 |
| random_agent | baseline | 0.2026 | 0.2446 | 0.8779 | 1.6480 | -0.3702 | 0.5000 |
| equal_weight | baseline | 0.2359 | 0.1921 | 1.2084 | 1.7274 | -0.2191 | 0.0000 |
| momentum | baseline | 0.1871 | 0.2095 | 0.9307 | 1.2379 | -0.4565 | 0.1540 |
| volatility_weighted | baseline | 0.2197 | 0.1494 | 1.4150 | 2.1367 | -0.2059 | 0.0273 |
| sp500 | baseline | 0.1611 | 0.1672 | 0.9824 | 1.5031 | -0.2480 | 0.0000 |
| sp500_gold_50_50 | baseline | 0.1774 | 0.1228 | 1.4007 | 2.3907 | -0.1740 | 0.0000 |

## Model Minus Baseline
Positive deltas are good for CAGR, Sharpe, Sortino, and maximum drawdown.
Negative deltas are good for volatility and turnover.

| baseline | model_minus_cagr | model_minus_volatility | model_minus_sharpe | model_minus_sortino | model_minus_maximum_drawdown | model_minus_average_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| random_agent | -0.0188 | -0.0376 | 0.0455 | -0.2239 | 0.0808 | 0.0466 |
| equal_weight | -0.0522 | 0.0149 | -0.2850 | -0.3033 | -0.0703 | 0.5466 |
| momentum | -0.0033 | -0.0025 | -0.0073 | 0.1862 | 0.1670 | 0.3926 |
| volatility_weighted | -0.0359 | 0.0576 | -0.4915 | -0.7126 | -0.0835 | 0.5194 |
| sp500 | 0.0227 | 0.0397 | -0.0590 | -0.0790 | -0.0415 | 0.5466 |
| sp500_gold_50_50 | 0.0064 | 0.0842 | -0.4773 | -0.9666 | -0.1154 | 0.5466 |

## Model Rank
| index | metric | direction | model_value | model_rank | strategies_compared | best_strategy |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | cagr | higher | 0.1838 | 5 | 7 | equal_weight |
| 1 | volatility | lower | 0.2070 | 5 | 7 | sp500_gold_50_50 |
| 2 | sharpe | higher | 0.9234 | 6 | 7 | volatility_weighted |
| 3 | sortino | higher | 1.4241 | 6 | 7 | sp500_gold_50_50 |
| 4 | maximum_drawdown | higher | -0.2894 | 5 | 7 | sp500_gold_50_50 |
| 5 | average_turnover | lower | 0.5466 | 7 | 7 | equal_weight |
