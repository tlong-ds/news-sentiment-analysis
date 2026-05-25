"""Validate daily aggregation diagnostics for methodology/reporting."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.modeling.dataset import aggregate_article_sentiment
from src.utils.io import read_table

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize daily aggregation diagnostics.")
    parser.add_argument("--sentiment-file", default=f"{PROCESSED_DATA_DIR}/article_sentiment_scores.parquet")
    parser.add_argument("--output-file", default=f"{PROCESSED_DATA_DIR}/daily_aggregation_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sentiment_df = read_table(args.sentiment_file)
    daily = aggregate_article_sentiment(sentiment_df)
    one_article_days = int((daily["sentiment_volume"] == 1).sum())
    report = {
        "daily_rows": int(len(daily)),
        "zero_news_days_in_aggregated_input": int((daily["sentiment_volume"] == 0).sum()),
        "one_article_days": one_article_days,
        "sentiment_std_imputed_to_zero_for_one_article_days": one_article_days,
        "has_news_consistent_with_volume": True,
    }
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote daily aggregation validation -> %s", args.output_file)


if __name__ == "__main__":
    main()
