"""VIX-based market regime classification."""

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class MarketRegime(StrEnum):
    """Volatility regimes ordered from low to high."""

    LOW_VOL = "LOW_VOL"
    MEDIUM_VOL = "MEDIUM_VOL"
    HIGH_VOL = "HIGH_VOL"


REGIME_ORDER: tuple[MarketRegime, ...] = tuple(MarketRegime)


@dataclass(frozen=True)
class VixRegimeModel:
    """Past-data VIX thresholds used to classify market regimes."""

    lower_threshold: float
    upper_threshold: float
    lower_quantile: float = 1.0 / 3.0
    upper_quantile: float = 2.0 / 3.0

    @classmethod
    def fit(
        cls,
        vix: pd.Series,
        quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    ) -> "VixRegimeModel":
        """Estimate fixed VIX terciles from one training series."""
        clean_vix = vix.dropna().astype(float)
        if clean_vix.empty:
            raise ValueError("vix training series cannot be empty")
        if (clean_vix <= 0.0).any():
            raise ValueError("vix training values must be positive")
        lower_quantile, upper_quantile = quantiles
        if not 0.0 < lower_quantile < upper_quantile < 1.0:
            raise ValueError("vix quantiles must be ordered inside zero and one")
        return cls(
            lower_threshold=float(clean_vix.quantile(lower_quantile)),
            upper_threshold=float(clean_vix.quantile(upper_quantile)),
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    def classify(self, vix: pd.Series) -> pd.Series:
        """Classify VIX levels with thresholds fitted on training data."""
        if vix.isna().any():
            raise ValueError("vix values cannot contain missing observations")
        labels = pd.Series(index=vix.index, dtype="object", name="regime")
        labels.loc[vix <= self.lower_threshold] = MarketRegime.LOW_VOL
        labels.loc[(vix > self.lower_threshold) & (vix <= self.upper_threshold)] = (
            MarketRegime.MEDIUM_VOL
        )
        labels.loc[vix > self.upper_threshold] = MarketRegime.HIGH_VOL
        return labels.astype(str)
