"""Evaluate the selected compact synthetic model and produce final artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR.parents[1] / "src"))


def main() -> None:
    """Run validation/test comparisons, policy inspection, and August forecast."""
    from agents.q_learning import QLearningAgent
    from agents.training import QLearningTrainingConfig
    from analysis.policy import (
        build_compact_policy_table,
        build_fixed_market_policy_table,
        build_reference_regime_policy,
        build_state_visit_summary,
        write_fixed_market_policy_svg,
        write_reference_policy_svg,
    )
    from backtesting.benchmarks import run_benchmarks
    from backtesting.engine import run_weight_strategy_backtest
    from backtesting.splitter import generate_walk_forward_splits
    from backtesting.strategies import (
        equal_weight_strategy,
        momentum_strategy,
        volatility_weighted_strategy,
    )
    from backtesting.walk_forward import run_walk_forward_train_validation_test
    from data.loaders import load_config
    from evaluation.artifacts import (
        build_normalized_value_paths,
        write_cumulative_return_svg,
    )
    from evaluation.report import build_performance_report
    from features.state_builder import PortfolioContext, StateBuilder
    from prediction.next_allocation import predict_next_allocations

    global RUN_DIR
    args = _parse_args()
    config_payload = _load_selected_config(Path(args.config))
    run_label = str(config_payload.pop("run_label", "selected_model"))
    RUN_DIR = Path(args.output_dir or Path("analysis/runs") / run_label)
    feature_mode = config_payload.pop("feature_mode")
    expected_splits = config_payload.pop("walk_forward_splits")
    training_config = QLearningTrainingConfig(**config_payload)
    project_config = load_config()
    dataset = pd.read_parquet(project_config["processed_dataset_path"])
    print(
        "Loaded selected configuration "
        f"({training_config.training_mode}, {training_config.episodes:,} episodes); "
        f"dataset through {dataset.index.max():%Y-%m-%d}",
        flush=True,
    )
    if not args.skip_full_training and not args.refresh_policy_analysis:
        _validate_prediction_target(dataset, args.target_period)
    splits = generate_walk_forward_splits(dataset.index)
    if len(splits) != expected_splits:
        raise ValueError(
            f"selected configuration expects {expected_splits} walk-forward splits, "
            f"but the dataset produces {len(splits)}",
        )

    walk_forward_dir = RUN_DIR / "walk_forward"
    comparison_dir = RUN_DIR / "comparison"
    analysis_dir = RUN_DIR / "policy"
    for directory in (walk_forward_dir, comparison_dir, analysis_dir):
        directory.mkdir(parents=True, exist_ok=True)
    reporter = RunReporter(RUN_DIR)
    reporter.update(
        stage="initialized",
        message="Selected-model production run initialized",
        training_mode=training_config.training_mode,
        episodes=training_config.episodes,
        split_count=len(splits),
        target_period=args.target_period,
    )

    if args.refresh_policy_analysis:
        reporter.update("policy_refresh", "Refreshing policy artifacts")
        model_path = RUN_DIR / "forecast" / args.target_period / "trained_model.json"
        agent = QLearningAgent.load(model_path)
        market_state = StateBuilder(dataset, feature_mode=feature_mode).build_state(
            len(dataset) - 1,
            PortfolioContext(weights=[0.25, 0.25, 0.25, 0.25]),
        )
        _write_policy_artifacts(
            agent=agent,
            analysis_dir=analysis_dir,
            build_compact_policy_table=build_compact_policy_table,
            build_state_visit_summary=build_state_visit_summary,
            build_reference_regime_policy=build_reference_regime_policy,
            write_reference_policy_svg=write_reference_policy_svg,
            build_fixed_market_policy_table=build_fixed_market_policy_table,
            write_fixed_market_policy_svg=write_fixed_market_policy_svg,
            market_state=market_state,
            actual_starting_weights=[0.25, 0.25, 0.25, 0.25],
            signal_date=f"{dataset.index[-1]:%Y-%m-%d}",
        )
        print(f"Refreshed policy artifacts: {analysis_dir}")
        reporter.complete("Policy artifacts refreshed")
        return

    if args.refresh_comparisons:
        reporter.update("comparison_refresh", "Refreshing baseline comparisons")
        _refresh_comparisons(
            dataset=dataset,
            splits=splits,
            comparison_dir=comparison_dir,
            run_benchmarks=run_benchmarks,
            run_weight_strategy_backtest=run_weight_strategy_backtest,
            strategies={
                "equal_weight": equal_weight_strategy,
                "momentum": momentum_strategy,
                "inverse_volatility": volatility_weighted_strategy,
            },
            build_performance_report=build_performance_report,
        )
        print(f"Refreshed comparison artifacts: {comparison_dir}")
        reporter.complete("Baseline comparisons refreshed")
        return

    print("Running walk-forward validation and test evaluation…", flush=True)
    reporter.update("walk_forward", "Starting validation and test evaluation")
    report, model_paths = run_walk_forward_train_validation_test(
        dataset=dataset,
        splits=splits,
        training_config=training_config,
        feature_mode=feature_mode,
        progress_callback=reporter.walk_forward_event,
    )
    report.to_csv(
        walk_forward_dir / "walk_forward_train_validation_test.csv", index=False
    )
    report.to_csv(comparison_dir / "walk_forward_model_metrics.csv", index=False)
    reporter.update("comparisons", "Writing baseline comparisons and charts")
    _write_comparisons(
        dataset=dataset,
        splits=splits,
        model_paths=model_paths,
        comparison_dir=comparison_dir,
        run_benchmarks=run_benchmarks,
        run_weight_strategy_backtest=run_weight_strategy_backtest,
        strategies={
            "equal_weight": equal_weight_strategy,
            "momentum": momentum_strategy,
            "inverse_volatility": volatility_weighted_strategy,
        },
        build_performance_report=build_performance_report,
        build_normalized_value_paths=build_normalized_value_paths,
        write_cumulative_return_svg=write_cumulative_return_svg,
    )

    if args.skip_full_training:
        _write_metadata(
            walk_forward_dir,
            comparison_dir,
            dataset,
            training_config,
            feature_mode,
        )
        print(f"Walk-forward artifacts: {walk_forward_dir}")
        reporter.complete("Walk-forward-only run completed")
        return

    print("Training full-data Model B for the August forecast…", flush=True)
    reporter.update("full_data_training", "Training full-data model for forecast")
    predictions, agent, history = predict_next_allocations(
        dataset=dataset,
        training_config=training_config,
        current_weights=[0.25, 0.25, 0.25, 0.25],
        feature_mode=feature_mode,
    )
    prediction = next(
        item for item in predictions if item.method == "model_b_modal_market_policy"
    )
    prediction_dir = RUN_DIR / "forecast" / prediction.target_period
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.to_dict() for item in predictions]).to_csv(
        prediction_dir / "allocation_predictions.csv",
        index=False,
    )
    agent.save(prediction_dir / "trained_model.json")
    history.to_csv(prediction_dir / "training_history.csv", index=False)
    print("Writing Model B policy and visit-coverage artifacts…", flush=True)
    reporter.update("policy_artifacts", "Writing policy and visit-coverage artifacts")
    _write_policy_artifacts(
        agent=agent,
        analysis_dir=analysis_dir,
        build_compact_policy_table=build_compact_policy_table,
        build_state_visit_summary=build_state_visit_summary,
        build_reference_regime_policy=build_reference_regime_policy,
        write_reference_policy_svg=write_reference_policy_svg,
        build_fixed_market_policy_table=build_fixed_market_policy_table,
        write_fixed_market_policy_svg=write_fixed_market_policy_svg,
        market_state=StateBuilder(dataset, feature_mode=feature_mode).build_state(
            len(dataset) - 1,
            PortfolioContext(weights=[0.25, 0.25, 0.25, 0.25]),
        ),
        actual_starting_weights=[0.25, 0.25, 0.25, 0.25],
        signal_date=f"{dataset.index[-1]:%Y-%m-%d}",
    )
    _write_metadata(
        walk_forward_dir,
        comparison_dir,
        dataset,
        training_config,
        feature_mode,
    )
    (prediction_dir / "prediction_metadata.json").write_text(
        json.dumps(
            {
                "training_config": asdict(training_config),
                "feature_mode": feature_mode,
                "dataset_end": f"{dataset.index.max():%Y-%m-%d}",
                "target_period": prediction.target_period,
                "policy_artifacts": str(analysis_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Walk-forward artifacts: {walk_forward_dir}")
    print(f"Comparison artifacts: {comparison_dir}")
    print(f"Policy artifacts: {analysis_dir}")
    print(f"Prediction: {prediction_dir / 'allocation_predictions.csv'}")
    reporter.complete("Selected-model production run completed")


class RunReporter:
    """Persist human-readable and machine-readable progress for long runs."""

    def __init__(self, run_dir: Path) -> None:
        self.status_path = run_dir / "status.json"
        self.log_path = run_dir / "run.log"

    def walk_forward_event(self, stage: str, split_id: int, message: str) -> None:
        """Receive structured updates from the walk-forward evaluator."""
        self.update(stage, message, split_id=split_id)

    def update(self, stage: str, message: str, **details: object) -> None:
        """Replace status and append an event to the run log."""
        timestamp = datetime.now(UTC).isoformat()
        status = {
            "status": "running",
            "updated_at": timestamp,
            "stage": stage,
            "message": message,
            **details,
        }
        self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} [{stage}] {message}\n")

    def complete(self, message: str) -> None:
        """Mark the run complete in both progress artifacts."""
        timestamp = datetime.now(UTC).isoformat()
        status = {
            "status": "completed",
            "updated_at": timestamp,
            "stage": "completed",
            "message": message,
        }
        self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} [completed] {message}\n")


def _write_comparisons(**kwargs: object) -> None:
    """Compare each real test window with passive and rule-based baselines."""
    dataset = kwargs["dataset"]
    splits = kwargs["splits"]
    model_paths = kwargs["model_paths"]
    comparison_dir = kwargs["comparison_dir"]
    for split in splits:
        model = model_paths[f"test_split_{split.split_id}"]
        test_data = dataset.loc[model.index]
        results = {"q_learning_synthetic": model}
        results.update(
            kwargs["run_benchmarks"](test_data[["r_arg", "r_ced", "r_sp500", "r_gold"]])
        )
        full_history = dataset.loc[: split.test_end]
        for name, strategy in kwargs["strategies"].items():
            baseline = kwargs["run_weight_strategy_backtest"](full_history, strategy)
            results[name] = _rebase_result_to_window(
                baseline.loc[baseline.index.intersection(model.index)]
            )
        metrics = kwargs["build_performance_report"](results)
        metrics.insert(
            0,
            "final_portfolio_value",
            [float(result["portfolio_value"].iloc[-1]) for result in results.values()],
        )
        metrics.to_csv(comparison_dir / f"test_split_{split.split_id}_metrics.csv")
        paths = kwargs["build_normalized_value_paths"](results)
        paths.to_csv(
            comparison_dir / f"test_split_{split.split_id}_cumulative_paths.csv"
        )
        kwargs["write_cumulative_return_svg"](
            paths,
            comparison_dir / f"test_split_{split.split_id}_cumulative_returns.svg",
            f"Historical test comparison — split {split.split_id}",
        )


def _refresh_comparisons(**kwargs: object) -> None:
    """Rebuild baseline metrics from saved test dates without retraining Q-learning."""
    dataset = kwargs["dataset"]
    splits = kwargs["splits"]
    comparison_dir = kwargs["comparison_dir"]
    model_metrics = pd.read_csv(comparison_dir / "walk_forward_model_metrics.csv")
    for split in splits:
        paths = pd.read_csv(
            comparison_dir / f"test_split_{split.split_id}_cumulative_paths.csv",
            index_col="date",
            parse_dates=True,
        )
        test_dates = paths.index
        test_data = dataset.loc[test_dates]
        baseline_results = kwargs["run_benchmarks"](
            test_data[["r_arg", "r_ced", "r_sp500", "r_gold"]]
        )
        full_history = dataset.loc[: split.test_end]
        for name, strategy in kwargs["strategies"].items():
            continuous_result = kwargs["run_weight_strategy_backtest"](
                full_history, strategy
            )
            baseline_results[name] = _rebase_result_to_window(
                continuous_result.loc[continuous_result.index.intersection(test_dates)]
            )
        baseline_metrics = kwargs["build_performance_report"](baseline_results)
        baseline_metrics.insert(
            0,
            "final_portfolio_value",
            [
                float(result["portfolio_value"].iloc[-1])
                for result in baseline_results.values()
            ],
        )
        model_row = model_metrics.loc[
            (model_metrics["split_id"] == split.split_id)
            & (model_metrics["phase"] == "test"),
            [
                "final_portfolio_value",
                "cagr",
                "volatility",
                "sharpe",
                "sortino",
                "maximum_drawdown",
                "average_turnover",
            ],
        ].copy()
        model_row.index = ["q_learning_synthetic"]
        pd.concat([model_row, baseline_metrics]).to_csv(
            comparison_dir / f"test_split_{split.split_id}_metrics.csv"
        )


def _rebase_result_to_window(result: pd.DataFrame) -> pd.DataFrame:
    """Reset value and drawdown to the beginning of one evaluation window."""
    if result.empty:
        raise ValueError("cannot rebase an empty backtest result")
    rebased = result.copy()
    return_column = (
        "net_portfolio_return"
        if "net_portfolio_return" in rebased.columns
        else "portfolio_return"
    )
    rebased["portfolio_value"] = (1.0 + rebased[return_column]).cumprod()
    rebased["drawdown"] = (
        rebased["portfolio_value"] / rebased["portfolio_value"].cummax() - 1.0
    )
    return rebased


def _write_policy_artifacts(**kwargs: object) -> None:
    """Persist greedy decisions and state-level training coverage diagnostics."""
    agent = kwargs["agent"]
    analysis_dir = kwargs["analysis_dir"]
    state_visits = kwargs["build_state_visit_summary"](agent)
    state_visits.to_csv(analysis_dir / "state_visit_summary.csv", index=False)
    policy = kwargs["build_compact_policy_table"](agent)
    policy.to_csv(analysis_dir / "compact_policy_all_states.csv", index=False)
    reference_policy = kwargs["build_reference_regime_policy"](policy)
    reference_policy.to_csv(
        analysis_dir / "compact_policy_balanced_reference.csv", index=False
    )
    kwargs["write_reference_policy_svg"](
        reference_policy, analysis_dir / "compact_policy_balanced_reference.svg"
    )
    fixed_market_policy = kwargs["build_fixed_market_policy_table"](
        agent,
        list(kwargs["market_state"]),
        list(kwargs["actual_starting_weights"]),
    )
    fixed_market_policy.to_csv(
        analysis_dir / "current_market_policy_by_starting_allocation.csv",
        index=False,
    )
    kwargs["write_fixed_market_policy_svg"](
        fixed_market_policy,
        list(kwargs["market_state"]),
        str(kwargs["signal_date"]),
        analysis_dir / "current_market_policy_by_starting_allocation.svg",
    )


def _load_selected_config(path: Path) -> dict[str, object]:
    """Read the committed fixed configuration rather than mutable HPO output."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selected configuration must be a YAML mapping")
    return payload


def _write_metadata(
    walk_forward_dir: Path,
    comparison_dir: Path,
    dataset: pd.DataFrame,
    training_config: object,
    feature_mode: str,
) -> None:
    """Persist the protocol and data boundary used for this final run."""
    metadata = {
        "dataset_start": f"{dataset.index.min():%Y-%m-%d}",
        "dataset_end": f"{dataset.index.max():%Y-%m-%d}",
        "feature_mode": feature_mode,
        "training_config": asdict(training_config),
        "protocol": (
            "Validation: fit synthetic simulator and Q-learning on train only. "
            "Test: refit a new simulator and Q-learning agent on train plus validation. "
            "Both evaluations use only historical observations and a frozen greedy policy."
        ),
        "overlap_note": "Test windows overlap, so charts and metrics are emitted per split rather than concatenated.",
    }
    for directory in (walk_forward_dir, comparison_dir):
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


def _validate_prediction_target(dataset: pd.DataFrame, target_period: str) -> None:
    """Fail before expensive training when the requested forecast is impossible."""
    expected_target = (
        pd.Timestamp(dataset.index.max()) + pd.offsets.MonthEnd(1)
    ).strftime("%Y-%m")
    if expected_target != target_period:
        raise ValueError(
            f"dataset ends at {dataset.index.max():%Y-%m}; it produces target "
            f"{expected_target}, not requested {target_period}. Refresh the monthly "
            "dataset before generating this forecast."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/selected_synthetic_model.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Self-contained destination; defaults to analysis/runs/<run_label>.",
    )
    parser.add_argument("--target-period", default="2026-08")
    parser.add_argument("--skip-full-training", action="store_true")
    parser.add_argument("--refresh-comparisons", action="store_true")
    parser.add_argument("--refresh-policy-analysis", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
