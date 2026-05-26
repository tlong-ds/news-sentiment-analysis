"""Robustness checks runner for the VN-Index volatility experiment."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.dataset import load_or_build_model_frame
from src.modeling.hybrid import (
    add_garch_features,
    build_lstm_sequences,
    evaluate_forecasts,
    train_lstm_residual_model,
    fit_garch11_baseline,
    fit_garchx11_baseline,
    fit_expanding_garch,
    diebold_mariano_test,
)
from src.tracking import (
    add_tracking_arguments,
    build_run_tags,
    collect_cli_params,
    configure_tracking,
    git_commit,
    tracking_config_from_args,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robustness checks for GARCH plus sentiment-LSTM volatility model."
    )
    parser.add_argument("--prices", default="data/raw/prices_VN.csv")
    parser.add_argument("--model-frame", default="data/interim/modeling_ready.parquet")
    parser.add_argument(
        "--daily-news", default="data/interim/daily_news_prices.parquet"
    )
    parser.add_argument(
        "--sentiment", default="data/sentiment/article_sentiment_scores.parquet"
    )
    parser.add_argument(
        "--articles-clean", default="data/interim/articles_clean.parquet"
    )
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--val-end", default="2023-12-31")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output", default="data/interim/robustness_experiment_summary.json"
    )
    add_tracking_arguments(parser)

    return parser.parse_args()


def run_spec(
    args: argparse.Namespace,
    sentiment_threshold: float = 0.05,
    target_type: str = "parkinson",
    use_sentiment_features: bool = True,
) -> dict[str, float]:
    """Build dataset and train/evaluate the GARCH + LSTM model for a specific robustness option."""
    # 1. Build model frame
    model_df = load_or_build_model_frame(
        model_frame_path=args.model_frame,
        price_path=args.prices,
        daily_news_path=args.daily_news,
        sentiment_path=args.sentiment,
        articles_clean_path=args.articles_clean,
        sentiment_threshold=sentiment_threshold,
        target_type=target_type,
    )
    model_df = add_garch_features(model_df, train_end=args.train_end)

    # 2. Select features
    if use_sentiment_features:
        feature_columns = [
            "garch_std_resid",
            "garch_forecast_vol",
            "abs_return",
            "n_articles",
            "n_categories",
            "mean_body_len",
            "mean_sentiment",
            "sentiment_std",
            "sentiment_volume",
            "negative_share",
            "neutral_share",
            "positive_share",
            "net_sentiment",
            "sentiment_surprise",
            "macro_sentiment",
            "market_sentiment",
            "has_sentiment",
        ]
    else:
        # Ablation study: market / news-intensity controls only, no sentiment
        feature_columns = [
            "garch_std_resid",
            "garch_forecast_vol",
            "abs_return",
            "n_articles",
            "n_categories",
            "mean_body_len",
        ]

    available_features = [col for col in feature_columns if col in model_df.columns]

    # 3. Create sequences
    sequences, _ = build_lstm_sequences(
        model_df,
        feature_columns=available_features,
        target_column="hybrid_residual_target",
        sequence_length=args.sequence_length,
        split_dates=(args.train_end, args.val_end),
    )

    baseline = sequences["baseline_test"]
    actual = sequences["realized_test"]

    # 4. Train LSTM
    lstm_model, _ = train_lstm_residual_model(
        sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # 5. Predict and evaluate
    residual_pred = (
        lstm_model(sequences["x_test"], training=False).numpy().reshape(-1, 1)
    )
    hybrid_pred = baseline + residual_pred

    baseline_metrics = evaluate_forecasts(actual, baseline)
    hybrid_metrics = evaluate_forecasts(actual, hybrid_pred)

    dm_stat, dm_pvalue = diebold_mariano_test(actual, baseline, hybrid_pred)

    return {
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_mape": baseline_metrics["mape"],
        "hybrid_rmse": hybrid_metrics["rmse"],
        "hybrid_mae": hybrid_metrics["mae"],
        "hybrid_mape": hybrid_metrics["mape"],
        "dm_stat": dm_stat,
        "dm_pvalue": dm_pvalue,
    }


def run_garchx(args: argparse.Namespace) -> dict[str, float]:
    """Fit and evaluate GARCH-X model where lagged sentiment is in the variance equation."""
    model_df = load_or_build_model_frame(
        model_frame_path=args.model_frame,
        price_path=args.prices,
        daily_news_path=args.daily_news,
        sentiment_path=args.sentiment,
        articles_clean_path=args.articles_clean,
    )

    # Exogenous variable: lagged mean sentiment
    model_df["lag_mean_sentiment"] = model_df["mean_sentiment"].shift(1).fillna(0.0)
    model_df = model_df.dropna(subset=["log_return", "target_next_vol"]).reset_index(
        drop=True
    )

    # Split index
    test_start = pd.Timestamp(args.val_end)
    train_df = model_df[pd.to_datetime(model_df["date"]) <= test_start]

    # Fit GARCH-X on training set
    garchx_fit = fit_garchx11_baseline(
        train_df["log_return"], exog=train_df["lag_mean_sentiment"]
    )

    # Project on full set recursively using fitted parameters
    y = model_df["log_return"].to_numpy() * 100.0
    x_exog = model_df["lag_mean_sentiment"].to_numpy()
    omega = garchx_fit.omega
    alpha = garchx_fit.alpha
    beta = garchx_fit.beta
    gamma = garchx_fit.gamma

    sample_var = float(np.var(y, ddof=1))
    sigma2 = np.empty_like(y)
    sigma2[0] = max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))
    for idx in range(1, len(y)):
        sigma2[idx] = (
            omega
            + alpha * y[idx - 1] ** 2
            + beta * sigma2[idx - 1]
            + gamma * x_exog[idx - 1]
        )

    # Forecast vol
    forecast_vol = np.empty_like(sigma2)
    forecast_vol[0] = (
        np.sqrt(max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))) / 100.0
    )
    for idx in range(1, len(y)):
        forecast_vol[idx] = (
            np.sqrt(
                omega
                + alpha * y[idx - 1] ** 2
                + beta * sigma2[idx - 1]
                + gamma * x_exog[idx - 1]
            )
            / 100.0
        )

    model_df["garchx_forecast_vol"] = forecast_vol

    # Evaluate on test set
    test_eval_df = model_df[pd.to_datetime(model_df["date"]) > test_start].copy()
    actual = test_eval_df["target_next_vol"].to_numpy()
    pred_garchx = test_eval_df["garchx_forecast_vol"].to_numpy()

    # Fit GARCH baseline for comparison
    garch_fit = fit_garch11_baseline(train_df["log_return"])
    y_baseline = model_df["log_return"].to_numpy() * 100.0
    omega_b = garch_fit.omega
    alpha_b = garch_fit.alpha
    beta_b = garch_fit.beta
    sigma2_b = np.empty_like(y_baseline)
    sigma2_b[0] = max(sample_var, omega_b / max(1e-6, 1.0 - alpha_b - beta_b))
    for idx in range(1, len(y_baseline)):
        sigma2_b[idx] = (
            omega_b + alpha_b * y_baseline[idx - 1] ** 2 + beta_b * sigma2_b[idx - 1]
        )

    forecast_b = np.empty_like(sigma2_b)
    forecast_b[0] = (
        np.sqrt(max(sample_var, omega_b / max(1e-6, 1.0 - alpha_b - beta_b))) / 100.0
    )
    for idx in range(1, len(y_baseline)):
        forecast_b[idx] = (
            np.sqrt(
                omega_b
                + alpha_b * y_baseline[idx - 1] ** 2
                + beta_b * sigma2_b[idx - 1]
            )
            / 100.0
        )

    model_df["garch_forecast_vol"] = forecast_b
    test_eval_df = model_df[pd.to_datetime(model_df["date"]) > test_start].copy()
    pred_garch = test_eval_df["garch_forecast_vol"].to_numpy()

    garch_metrics = evaluate_forecasts(actual, pred_garch)
    garchx_metrics = evaluate_forecasts(actual, pred_garchx)
    dm_stat, dm_pvalue = diebold_mariano_test(actual, pred_garch, pred_garchx)

    return {
        "baseline_rmse": garch_metrics["rmse"],
        "baseline_mae": garch_metrics["mae"],
        "baseline_mape": garch_metrics["mape"],
        "hybrid_rmse": garchx_metrics["rmse"],
        "hybrid_mae": garchx_metrics["mae"],
        "hybrid_mape": garchx_metrics["mape"],
        "dm_stat": dm_stat,
        "dm_pvalue": dm_pvalue,
        "garchx_gamma": gamma,
    }


def run_expanding_garch_eval(args: argparse.Namespace) -> dict[str, float]:
    """Evaluate GARCH + LSTM model where GARCH parameters are re-estimated expanding-window-style."""
    model_df = load_or_build_model_frame(
        model_frame_path=args.model_frame,
        price_path=args.prices,
        daily_news_path=args.daily_news,
        sentiment_path=args.sentiment,
        articles_clean_path=args.articles_clean,
    )

    # Split index
    test_start = pd.Timestamp(args.val_end)
    train_val_df = model_df[pd.to_datetime(model_df["date"]) <= test_start]
    train_len = len(train_val_df[train_val_df["log_return"].notna()])

    # expanding GARCH fitting
    cond_vol, forecast_vol, std_resid = fit_expanding_garch(
        model_df["log_return"],
        train_len=train_len,
        reestimate_freq=21,
    )

    fitted = pd.Series(index=model_df.index, dtype=float)
    forecast = pd.Series(index=model_df.index, dtype=float)
    zscore = pd.Series(index=model_df.index, dtype=float)

    valid_index = model_df[model_df["log_return"].notna()].index
    fitted.loc[valid_index] = cond_vol
    forecast.loc[valid_index] = forecast_vol
    zscore.loc[valid_index] = std_resid

    model_df["garch_conditional_vol"] = fitted
    model_df["garch_forecast_vol"] = forecast
    model_df["garch_std_resid"] = zscore
    model_df["hybrid_residual_target"] = (
        model_df["target_next_vol"] - model_df["garch_forecast_vol"]
    )

    feature_columns = [
        "garch_std_resid",
        "garch_forecast_vol",
        "abs_return",
        "n_articles",
        "n_categories",
        "mean_body_len",
        "mean_sentiment",
        "sentiment_std",
        "sentiment_volume",
        "negative_share",
        "neutral_share",
        "positive_share",
        "net_sentiment",
        "sentiment_surprise",
        "macro_sentiment",
        "market_sentiment",
        "has_sentiment",
    ]
    available_features = [col for col in feature_columns if col in model_df.columns]

    # Create sequences
    sequences, _ = build_lstm_sequences(
        model_df,
        feature_columns=available_features,
        target_column="hybrid_residual_target",
        sequence_length=args.sequence_length,
        split_dates=(args.train_end, args.val_end),
    )

    baseline = sequences["baseline_test"]
    actual = sequences["realized_test"]

    # Train LSTM
    lstm_model, _ = train_lstm_residual_model(
        sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Predict and evaluate
    residual_pred = (
        lstm_model(sequences["x_test"], training=False).numpy().reshape(-1, 1)
    )
    hybrid_pred = baseline + residual_pred

    baseline_metrics = evaluate_forecasts(actual, baseline)
    hybrid_metrics = evaluate_forecasts(actual, hybrid_pred)
    dm_stat, dm_pvalue = diebold_mariano_test(actual, baseline, hybrid_pred)

    return {
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_mape": baseline_metrics["mape"],
        "hybrid_rmse": hybrid_metrics["rmse"],
        "hybrid_mae": hybrid_metrics["mae"],
        "hybrid_mape": hybrid_metrics["mape"],
        "dm_stat": dm_stat,
        "dm_pvalue": dm_pvalue,
    }


def main() -> None:
    """CLI entrypoint for robustness checks."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    tracking_config = tracking_config_from_args(args)
    tracking = configure_tracking(tracking_config)
    run_name = tracking_config.run_name or "run_robustness"

    with tracking.start_run(
        run_name=run_name,
        tags=build_run_tags(
            stage="run_robustness",
            pipeline_mode="modeling",
            source_dataset="cafef",
        ),
    ):
        tracking.log_params(collect_cli_params(args))
        tracking.log_params(
            {
                "invoked_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit(),
            }
        )
        results = {}
        specs = [
            (
                "baseline",
                "Running Baseline Model...",
                {
                    "sentiment_threshold": 0.05,
                    "target_type": "parkinson",
                    "use_sentiment_features": True,
                },
            ),
            (
                "threshold_0.10",
                "Running Alternative Threshold (0.10)...",
                {
                    "sentiment_threshold": 0.10,
                    "target_type": "parkinson",
                    "use_sentiment_features": True,
                },
            ),
            (
                "threshold_0.15",
                "Running Alternative Threshold (0.15)...",
                {
                    "sentiment_threshold": 0.15,
                    "target_type": "parkinson",
                    "use_sentiment_features": True,
                },
            ),
            (
                "garman_klass",
                "Running Garman-Klass Volatility target...",
                {
                    "sentiment_threshold": 0.05,
                    "target_type": "garman_klass",
                    "use_sentiment_features": True,
                },
            ),
            (
                "no_sentiment_ablation",
                "Running Ablation Study (No Sentiment)...",
                {
                    "sentiment_threshold": 0.05,
                    "target_type": "parkinson",
                    "use_sentiment_features": False,
                },
            ),
        ]

        for name, message, options in specs:
            logger.info(message)
            with tracking.start_run(
                run_name=name,
                nested=True,
                tags=build_run_tags(
                    stage=name,
                    pipeline_mode="modeling_robustness",
                    source_dataset="cafef",
                ),
            ):
                tracking.log_params(options)
                result = run_spec(args, **options)
                tracking.log_metrics(result)
                results[name] = result

        logger.info("Running GARCH-X Exogenous Sentiment Baseline...")
        with tracking.start_run(
            run_name="garch_x",
            nested=True,
            tags=build_run_tags(
                stage="garch_x",
                pipeline_mode="modeling_robustness",
                source_dataset="cafef",
            ),
        ):
            garch_x_result = run_garchx(args)
            tracking.log_metrics(garch_x_result)
            results["garch_x"] = garch_x_result

        logger.info("Running Expanding Window GARCH Re-estimation...")
        with tracking.start_run(
            run_name="expanding_garch",
            nested=True,
            tags=build_run_tags(
                stage="expanding_garch",
                pipeline_mode="modeling_robustness",
                source_dataset="cafef",
            ),
        ):
            expanding_result = run_expanding_garch_eval(args)
            tracking.log_metrics(expanding_result)
            results["expanding_garch"] = expanding_result

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        tracking.log_artifact(output_path)
        logger.info("Robustness results written to %s", args.output)

        print("\n" + "=" * 80)
        print("PHASE 7 ROBUSTNESS CHECKS COMPARISON TABLE")
        print("=" * 80)
        print(
            "| Specification | GARCH RMSE | Hybrid RMSE | GARCH MAE | Hybrid MAE | DM Stat | DM p-value |"
        )
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for name, result in results.items():
            print(
                f"| **{name}** | {result['baseline_rmse']:.6f} | {result['hybrid_rmse']:.6f} | "
                f"{result['baseline_mae']:.6f} | {result['hybrid_mae']:.6f} | "
                f"{result['dm_stat']:.4f} | {result['dm_pvalue']:.5f} |"
            )
        print("=" * 80)


if __name__ == "__main__":
    main()
