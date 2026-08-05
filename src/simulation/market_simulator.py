"""Regime-switching synthetic monthly market simulator."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simulation.markov_chain import RegimeMarkovChain
from simulation.multivariate_t import MultivariateStudentT
from simulation.regimes import REGIME_ORDER, MarketRegime, VixRegimeModel

SIMULATION_CONFIG_PATH = Path("src/config/simulation.yaml")
ASSET_KEYS = ("arg", "ced", "sp500", "gold")
RETURN_COLUMNS = tuple(f"r_{asset}" for asset in ASSET_KEYS)
PRICE_COLUMNS = tuple(f"price_{asset}" for asset in ASSET_KEYS)
PRIMITIVE_COLUMNS = (
    "log_return_arg",
    "log_return_ced",
    "log_return_sp500",
    "log_return_gold",
    "log_change_usd_ars",
    "log_change_vix",
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for fitting and streaming synthetic market episodes."""

    vix_quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    transition_smoothing: float = 1.0
    degrees_freedom: float = 8.0
    covariance_regularization: float = 1e-6
    episode_count: int = 1000
    episode_length: int = 120
    seed: int = 42
    min_vix: float = 8.0
    max_vix: float = 100.0
    max_abs_log_change: float = 1.0

    def validate(self) -> None:
        """Validate simulation parameters."""
        lower_quantile, upper_quantile = self.vix_quantiles
        if not 0.0 < lower_quantile < upper_quantile < 1.0:
            raise ValueError("vix quantiles must be ordered inside zero and one")
        if self.transition_smoothing < 0.0:
            raise ValueError("transition_smoothing must be non-negative")
        if self.degrees_freedom <= 2.0:
            raise ValueError("degrees_freedom must be greater than two")
        if self.covariance_regularization <= 0.0:
            raise ValueError("covariance_regularization must be positive")
        if self.episode_count < 1:
            raise ValueError("episode_count must be positive")
        if self.episode_length < 2:
            raise ValueError("episode_length must be at least two")
        if not 0.0 < self.min_vix < self.max_vix:
            raise ValueError("VIX limits must be positive and ordered")
        if self.max_abs_log_change <= 0.0:
            raise ValueError("max_abs_log_change must be positive")


def load_simulation_config(
    path: str | Path = SIMULATION_CONFIG_PATH,
) -> SimulationConfig:
    """Load the synthetic market configuration from YAML."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("simulation configuration must be a mapping")
    config = SimulationConfig(
        vix_quantiles=tuple(float(value) for value in payload["vix_quantiles"]),
        transition_smoothing=float(payload["transition_smoothing"]),
        degrees_freedom=float(payload["degrees_freedom"]),
        covariance_regularization=float(payload["covariance_regularization"]),
        episode_count=int(payload["episode_count"]),
        episode_length=int(payload["episode_length"]),
        seed=int(payload["seed"]),
        min_vix=float(payload["min_vix"]),
        max_vix=float(payload["max_vix"]),
        max_abs_log_change=float(payload["max_abs_log_change"]),
    )
    config.validate()
    return config


@dataclass(frozen=True)
class MarketSimulator:
    """Fitted Markov-regime multivariate Student-t market model."""

    config: SimulationConfig
    regime_model: VixRegimeModel
    markov_chain: RegimeMarkovChain
    regime_distributions: dict[MarketRegime, MultivariateStudentT]
    initial_prices: np.ndarray
    initial_usd_ars: float
    initial_vix: float
    training_start: pd.Timestamp
    training_end: pd.Timestamp

    @classmethod
    def fit(
        cls,
        train_dataset: pd.DataFrame,
        config: SimulationConfig | None = None,
    ) -> "MarketSimulator":
        """Fit the simulator using only the supplied training dataset."""
        simulation_config = config or load_simulation_config()
        simulation_config.validate()
        _validate_training_dataset(train_dataset)

        primitive_data = _build_primitive_data(train_dataset)
        regime_model = VixRegimeModel.fit(
            train_dataset["vix"],
            quantiles=simulation_config.vix_quantiles,
        )
        all_regimes = regime_model.classify(train_dataset["vix"])
        primitive_regimes = all_regimes.loc[primitive_data.index]
        markov_chain = RegimeMarkovChain.fit(
            all_regimes,
            smoothing=simulation_config.transition_smoothing,
        )

        global_covariance = np.atleast_2d(
            np.cov(primitive_data.to_numpy(), rowvar=False, ddof=1),
        )
        distributions: dict[MarketRegime, MultivariateStudentT] = {}
        for regime in REGIME_ORDER:
            regime_observations = primitive_data.loc[
                primitive_regimes == regime.value
            ].to_numpy()
            if len(regime_observations) == 0:
                regime_observations = primitive_data.to_numpy()
            distributions[regime] = MultivariateStudentT.fit(
                observations=regime_observations,
                degrees_freedom=simulation_config.degrees_freedom,
                regularization=simulation_config.covariance_regularization,
                fallback_covariance=global_covariance,
            )

        return cls(
            config=simulation_config,
            regime_model=regime_model,
            markov_chain=markov_chain,
            regime_distributions=distributions,
            initial_prices=train_dataset.loc[:, list(PRICE_COLUMNS)]
            .iloc[-1]
            .to_numpy(float),
            initial_usd_ars=float(train_dataset["usd_ars"].iloc[-1]),
            initial_vix=float(train_dataset["vix"].iloc[-1]),
            training_start=pd.Timestamp(train_dataset.index.min()),
            training_end=pd.Timestamp(train_dataset.index.max()),
        )

    def stream_episodes(self) -> Iterator[pd.DataFrame]:
        """Yield one synthetic episode at a time without retaining prior episodes."""
        master_rng = np.random.default_rng(self.config.seed)
        for _ in range(self.config.episode_count):
            episode_seed = int(master_rng.integers(0, np.iinfo(np.int64).max))
            yield self.simulate_episode(np.random.default_rng(episode_seed))

    def simulate_episode(
        self,
        rng: np.random.Generator | None = None,
        length: int | None = None,
    ) -> pd.DataFrame:
        """Simulate and reconstruct one monthly market dataset."""
        episode_rng = rng or np.random.default_rng(self.config.seed)
        episode_length = length or self.config.episode_length
        if episode_length < 2:
            raise ValueError("episode length must be at least two")

        regimes = self.markov_chain.sample_path(episode_length, episode_rng)
        primitives = np.vstack(
            [
                self.regime_distributions[regime].sample(episode_rng)
                for regime in regimes
            ]
        )
        primitives = np.clip(
            primitives,
            -self.config.max_abs_log_change,
            self.config.max_abs_log_change,
        )
        return self._reconstruct_dataset(primitives, regimes)

    def _reconstruct_dataset(
        self,
        primitives: np.ndarray,
        regimes: list[MarketRegime],
    ) -> pd.DataFrame:
        """Reconstruct positive levels and simple returns from log changes."""
        asset_log_returns = primitives[:, : len(ASSET_KEYS)]
        simple_returns = np.expm1(asset_log_returns)
        price_levels = self.initial_prices * np.exp(
            np.cumsum(asset_log_returns, axis=0)
        )
        usd_ars = self.initial_usd_ars * np.exp(np.cumsum(primitives[:, 4]))

        vix_values: list[float] = []
        current_vix = self.initial_vix
        for log_change in primitives[:, 5]:
            current_vix = float(
                np.clip(
                    current_vix * np.exp(log_change),
                    self.config.min_vix,
                    self.config.max_vix,
                ),
            )
            vix_values.append(current_vix)

        index = pd.date_range(
            self.training_end + pd.offsets.MonthEnd(1),
            periods=len(primitives),
            freq="ME",
            name="date",
        )
        data: dict[str, np.ndarray | list[float] | list[str]] = {}
        for column_index, column in enumerate(PRICE_COLUMNS):
            data[column] = price_levels[:, column_index]
        for column_index, column in enumerate(RETURN_COLUMNS):
            data[column] = simple_returns[:, column_index]
        data["usd_ars"] = usd_ars
        data["vix"] = vix_values
        episode = pd.DataFrame(data, index=index)
        episode.attrs["simulated_regimes"] = [regime.value for regime in regimes]
        episode.attrs["training_start"] = self.training_start.isoformat()
        episode.attrs["training_end"] = self.training_end.isoformat()
        return episode


def _build_primitive_data(train_dataset: pd.DataFrame) -> pd.DataFrame:
    """Build log returns and macro log changes from training observations."""
    if (train_dataset.loc[:, list(RETURN_COLUMNS)] <= -1.0).any().any():
        raise ValueError("asset returns must be greater than negative one")
    primitives = pd.DataFrame(index=train_dataset.index)
    for asset, return_column in zip(ASSET_KEYS, RETURN_COLUMNS, strict=True):
        primitives[f"log_return_{asset}"] = np.log1p(train_dataset[return_column])
    primitives["log_change_usd_ars"] = np.log(train_dataset["usd_ars"]).diff()
    primitives["log_change_vix"] = np.log(train_dataset["vix"]).diff()
    return primitives.loc[:, PRIMITIVE_COLUMNS].dropna()


def _validate_training_dataset(train_dataset: pd.DataFrame) -> None:
    """Validate the primitive columns required to fit the simulator."""
    required_columns = {*PRICE_COLUMNS, *RETURN_COLUMNS, "usd_ars", "vix"}
    missing_columns = required_columns - set(train_dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"training dataset is missing columns: {missing}")
    if len(train_dataset) < 3:
        raise ValueError("training dataset must contain at least three months")
    if not isinstance(train_dataset.index, pd.DatetimeIndex):
        raise ValueError("training dataset must use a DatetimeIndex")
    if not train_dataset.index.is_monotonic_increasing:
        raise ValueError("training dataset index must be sorted")
    if train_dataset.isna().any().any():
        raise ValueError("training dataset cannot contain missing values")
    level_columns = [*PRICE_COLUMNS, "usd_ars", "vix"]
    if (train_dataset.loc[:, level_columns] <= 0.0).any().any():
        raise ValueError("prices, USD/ARS, and VIX must be positive")
