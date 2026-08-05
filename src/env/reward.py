"""CRRA reward calculation for portfolio rebalancing decisions."""

from math import log

DEFAULT_RISK_AVERSION = 0.5


def calculate_reward(
    portfolio_return: float,
    turnover: float,
    transaction_cost: float = 0.014,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
) -> float:
    """Calculate CRRA utility of the monthly return net of trading costs.

    The reward is ``((1 + net_return) ** (1 - gamma) - 1) / (1 - gamma)``.
    The logarithmic limit is used when ``gamma`` equals one.
    """
    net_return = calculate_net_portfolio_return(
        portfolio_return=portfolio_return,
        turnover=turnover,
        transaction_cost=transaction_cost,
    )
    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be non-negative")
    gross_return = 1.0 + net_return
    if gross_return <= 0.0:
        raise ValueError("net portfolio return must be greater than negative one")
    if risk_aversion == 1.0:
        return log(gross_return)
    return (gross_return ** (1.0 - risk_aversion) - 1.0) / (1.0 - risk_aversion)


def calculate_net_portfolio_return(
    portfolio_return: float,
    turnover: float,
    transaction_cost: float = 0.014,
) -> float:
    """Return the portfolio return after rebalancing costs."""
    if turnover < 0.0:
        raise ValueError("turnover must be non-negative")
    if transaction_cost < 0.0:
        raise ValueError("transaction_cost must be non-negative")
    return portfolio_return - transaction_cost * turnover
