"""Generate discrete portfolio allocations."""


def generate_action_space(
    step: float = 0.10,
    n_assets: int = 4,
) -> list[list[float]]:
    """Return every non-negative allocation that sums to one.

    Parameters
    ----------
    step
        Difference between consecutive weight values. It must divide one
        exactly.
    n_assets
        Number of assets in each allocation.

    Returns
    -------
    list[list[float]]
        Portfolio weights ordered from highest to lowest weight in each
        successive asset.
    """
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be greater than zero and at most one")
    if n_assets < 1:
        raise ValueError("n_assets must be at least one")

    total_units = round(1.0 / step)
    if not _is_close(total_units * step, 1.0):
        raise ValueError("step must divide one exactly")

    allocations = _generate_unit_allocations(total_units, n_assets)
    return [
        [round(units / total_units, 10) for units in allocation]
        for allocation in allocations
    ]


def _generate_unit_allocations(total_units: int, n_assets: int) -> list[list[int]]:
    """Generate integer compositions of ``total_units`` into ``n_assets`` parts."""
    if n_assets == 1:
        return [[total_units]]

    allocations: list[list[int]] = []
    for first_weight in range(total_units, -1, -1):
        remaining_allocations = _generate_unit_allocations(
            total_units - first_weight,
            n_assets - 1,
        )
        allocations.extend(
            [first_weight, *remaining] for remaining in remaining_allocations
        )
    return allocations


def _is_close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    """Return whether two floats differ by no more than ``tolerance``."""
    return abs(left - right) <= tolerance
