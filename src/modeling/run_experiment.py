"""CLI entrypoint for the VN-Index hybrid volatility experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.modeling.dataset import build_model_frame
from src.modeling.hybrid import (
    add_garch_features,
    build_lstm_sequences,
    evaluate_forecasts,
    train_lstm_residual_model,
    fit_garch11_baseline,
    validate_garch_fit,
    diebold_mariano_test,
    analyze_forecast_subperiods,
)
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-stage GARCH plus sentiment-LSTM volatility experiment."
    )
    parser.add_argument("--prices", default="data/raw/prices_VN.csv")
    parser.add_argument("--daily-news", default="data/processed/daily_news_prices.csv")
    parser.add_argument(
        "--sentiment",
        default=None,
        help="CSV with article-level sentiment_score or daily sentiment aggregates.",
    )
    parser.add_argument(
        "--articles-clean",
        default="data/processed/articles_clean.csv",
        help="CSV with article-level metadata including categories (used to extract macro/market subsets).",
    )
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--val-end", default="2023-12-31")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build the experiment frame and evaluate the GARCH baseline without training the LSTM.",
    )
    parser.add_argument("--output", default="data/processed/hybrid_experiment_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_df = build_model_frame(
        args.prices,
        daily_news_path=args.daily_news,
        sentiment_path=args.sentiment,
        articles_clean_path=args.articles_clean,
    )
    model_df = add_garch_features(model_df, train_end=args.train_end)

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
    sequences, meta = build_lstm_sequences(
        model_df,
        feature_columns=available_features,
        target_column="hybrid_residual_target",
        sequence_length=args.sequence_length,
        split_dates=(args.train_end, args.val_end),
    )

    baseline = sequences["baseline_test"]
    actual = sequences["realized_test"]

    # Fit baseline GARCH directly on training set to compute diagnostic validation
    train_returns = model_df[pd.to_datetime(model_df["date"]) <= pd.Timestamp(args.train_end)]["log_return"]
    garch_result = fit_garch11_baseline(train_returns)
    garch_diagnostics = validate_garch_fit(garch_result)

    summary = {
        "feature_columns": meta.feature_columns,
        "sequence_length": meta.sequence_length,
        "split_sizes": {
            "train": meta.train_rows,
            "val": meta.validation_rows,
            "test": meta.test_rows,
        },
        "garch_diagnostics": garch_diagnostics,
        "baseline_metrics": evaluate_forecasts(actual, baseline),
    }

    if args.prepare_only:
        summary["status"] = "prepared_without_lstm"
    else:
        lstm_model, history = train_lstm_residual_model(
            sequences,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        residual_pred = lstm_model(sequences["x_test"], training=False).numpy().reshape(-1, 1)
        hybrid_pred = baseline + residual_pred
        summary["hybrid_metrics"] = evaluate_forecasts(actual, hybrid_pred)
        summary["history"] = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        summary["hybrid_forecast"] = [float(value) for value in hybrid_pred.reshape(-1)]

        # Run Diebold-Mariano test (Newey-West lag=1)
        dm_stat, dm_pvalue = diebold_mariano_test(actual, baseline, hybrid_pred)
        summary["diebold_mariano"] = {
            "statistic": dm_stat,
            "p_value": dm_pvalue,
            "significant_95": bool(dm_pvalue < 0.05),
        }

        # Run Subperiod and Asymmetry analysis
        test_dates = pd.to_datetime(sequences["dates_test"])
        test_sent_df = pd.DataFrame({"date": test_dates}).merge(
            model_df[["date", "mean_sentiment"]], on="date", how="left"
        )
        test_sentiment = test_sent_df["mean_sentiment"].to_numpy()

        subperiod_metrics = analyze_forecast_subperiods(
            actual=actual,
            pred_baseline=baseline,
            pred_hybrid=hybrid_pred,
            dates=sequences["dates_test"],
            sentiment=test_sentiment,
        )
        summary["subperiod_analysis"] = subperiod_metrics

    summary["test_dates"] = [str(date) for date in sequences["dates_test"]]
    summary["actual_volatility"] = [float(value) for value in actual.reshape(-1)]
    summary["baseline_forecast"] = [float(value) for value in baseline.reshape(-1)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    print("\n=== GARCH Baseline Diagnostics ===")
    print(json.dumps(summary["garch_diagnostics"], indent=2))
    print("\n=== Baseline Volatility Forecast Metrics ===")
    print(json.dumps(summary["baseline_metrics"], indent=2))
    if "hybrid_metrics" in summary:
        print("\n=== Hybrid Volatility Forecast Metrics ===")
        print(json.dumps(summary["hybrid_metrics"], indent=2))
        print("\n=== Diebold-Mariano Comparative Test ===")
        print(json.dumps(summary["diebold_mariano"], indent=2))
        print("\n=== Subperiod and Asymmetry Analysis ===")
        print(json.dumps(summary["subperiod_analysis"], indent=2))


if __name__ == "__main__":
    main()
