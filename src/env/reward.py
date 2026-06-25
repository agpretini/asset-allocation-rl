"""Reward calculation for portfolio rebalancing decisions."""


def calculate_reward(
    portfolio_return: float,
    turnover: float,
    transaction_cost: float = 0.014,
) -> float:
    """Calculate reward as portfolio return net of rebalancing cost."""
    if turnover < 0.0:
        raise ValueError("turnover must be non-negative")
    if transaction_cost < 0.0:
        raise ValueError("transaction_cost must be non-negative")
    return portfolio_return - transaction_cost * turnover
