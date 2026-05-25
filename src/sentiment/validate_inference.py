"""Validate article-level inference output before modeling."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.modeling.dataset import aggregate_article_sentiment
from src.utils.io import read_table

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate article-level sentiment inference output.")
    parser.add_argument("--articles-file", default=f"{PROCESSED_DATA_DIR}/articles_clean.parquet")
    parser.add_argument("--sentiment-file", default=f"{PROCESSED_DATA_DIR}/article_sentiment_scores.parquet")
    parser.add_argument("--daily-news-file", default=f"{PROCESSED_DATA_DIR}/daily_news_prices.parquet")
    parser.add_argument("--report-file", default=f"{PROCESSED_DATA_DIR}/sentiment_inference_validation.json")
    parser.add_argument("--fail-on-validation", action="store_true")
    return parser.parse_args()


def validate_outputs(articles_df: pd.DataFrame, sentiment_df: pd.DataFrame, daily_news_df: pd.DataFrame | None = None) -> dict:
    required_cols = {
        "url",
        "trading_date",
        "category",
        "sentiment_score",
        "sentiment_label",
        "prob_positive",
        "prob_negative",
        "prob_neutral",
    }
    missing = sorted(required_cols - set(sentiment_df.columns))
    if missing:
        raise ValueError(f"Sentiment output missing columns: {missing}")

    probability_sums = (
        sentiment_df[["prob_positive", "prob_negative", "prob_neutral"]]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )
    score_hist_counts, score_hist_edges = np.histogram(sentiment_df["sentiment_score"].to_numpy(dtype=float), bins=20, range=(-1.0, 1.0))
    year_distribution_df = sentiment_df.assign(year=pd.to_datetime(sentiment_df["trading_date"], errors="coerce").dt.year)
    year_distribution = year_distribution_df.pivot_table(index="year", columns="sentiment_label", values="url", aggfunc="count", fill_value=0)
    year_shares = year_distribution.div(year_distribution.sum(axis=1), axis=0).fillna(0.0)
    dominant_year_flags = {
        str(year): row[row > 0.70].to_dict()
        for year, row in year_shares.iterrows()
        if (row > 0.70).any()
    }
    category_distribution = sentiment_df.pivot_table(index="category", columns="sentiment_label", values="url", aggfunc="count", fill_value=0)
    category_shares = category_distribution.div(category_distribution.sum(axis=1), axis=0).fillna(0.0)
    stock_negative = float(category_shares.loc["Chứng khoán", "negative"]) if "Chứng khoán" in category_shares.index and "negative" in category_shares.columns else None
    business_negative = float(category_shares.loc["Kinh doanh", "negative"]) if "Kinh doanh" in category_shares.index and "negative" in category_shares.columns else None
    category_check_passed = (
        stock_negative is not None
        and business_negative is not None
        and stock_negative > business_negative
    )

    volatility_check = None
    if daily_news_df is not None and "log_return" in daily_news_df.columns:
        daily_sentiment = aggregate_article_sentiment(sentiment_df)
        merged = daily_news_df.copy()
        merged["abs_return"] = pd.to_numeric(merged["log_return"], errors="coerce").abs()
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.merge(daily_sentiment[["date", "negative_share"]], on="date", how="left")
        top20 = merged.nlargest(20, "abs_return")
        corpus_negative = float(daily_sentiment["negative_share"].mean()) if not daily_sentiment.empty else 0.0
        top20_negative = float(top20["negative_share"].fillna(0.0).mean()) if not top20.empty else 0.0
        volatility_check = {
            "corpus_negative_share_mean": corpus_negative,
            "top20_volatility_negative_share_mean": top20_negative,
            "passed": bool(top20_negative > corpus_negative),
        }

    diagnostics = {
        "articles_rows": int(len(articles_df)),
        "sentiment_rows": int(len(sentiment_df)),
        "row_count_match": bool(len(articles_df) == len(sentiment_df)),
        "duplicate_urls": int(sentiment_df["url"].duplicated().sum()),
        "probability_sum_max_abs_error": float(np.max(np.abs(probability_sums - 1.0))),
        "sentiment_score_histogram": {
            "counts": score_hist_counts.tolist(),
            "bin_edges": score_hist_edges.tolist(),
        },
        "label_distribution": sentiment_df["sentiment_label"].value_counts(normalize=True).to_dict(),
        "year_distribution": year_distribution.to_dict(),
        "category_distribution": category_distribution.to_dict(),
        "year_class_dominance_flags": dominant_year_flags,
        "category_negative_share_check": {
            "stock_negative_share": stock_negative,
            "business_negative_share": business_negative,
            "passed": category_check_passed,
        },
        "top20_volatility_day_check": volatility_check,
    }
    diagnostics["validation_checks"] = {
        "row_count_match": diagnostics["row_count_match"],
        "no_duplicate_urls": diagnostics["duplicate_urls"] == 0,
        "probabilities_sum_to_one": diagnostics["probability_sum_max_abs_error"] < 1e-5,
        "no_year_above_70pct_single_class": len(dominant_year_flags) == 0,
        "category_negative_share_check": bool(category_check_passed),
        "top20_volatility_day_check": volatility_check["passed"] if volatility_check is not None else False,
    }
    diagnostics["ready_for_modeling"] = all(diagnostics["validation_checks"].values())
    return diagnostics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    articles_df = read_table(args.articles_file)
    sentiment_df = read_table(args.sentiment_file)
    daily_news_df = read_table(args.daily_news_file) if Path(args.daily_news_file).exists() else None
    diagnostics = validate_outputs(articles_df, sentiment_df, daily_news_df=daily_news_df)
    with open(args.report_file, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)
    if args.fail_on_validation and not diagnostics["ready_for_modeling"]:
        raise RuntimeError(f"Sentiment inference validation failed. See {args.report_file}")
    logger.info("Wrote validation report -> %s", args.report_file)


if __name__ == "__main__":
    main()
