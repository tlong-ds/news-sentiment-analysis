"""Merge full CafeF article texts with inferred sentiment scores."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.utils.io import read_parquet_table

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge cleaned article texts with inferred sentiment scores."
    )
    parser.add_argument(
        "--articles-file",
        default=f"{PROCESSED_DATA_DIR}/articles_clean.parquet",
        help="Path to the cleaned article parquet file.",
    )
    parser.add_argument(
        "--sentiment-file",
        default=f"{PROCESSED_DATA_DIR}/article_sentiment_scores.parquet",
        help="Path to the inferred sentiment scores parquet file.",
    )
    parser.add_argument(
        "--output-file",
        default=f"{PROCESSED_DATA_DIR}/articles_with_sentiment.parquet",
        help="Path to save the merged parquet output.",
    )
    parser.add_argument(
        "--how",
        choices=["inner", "left"],
        default="inner",
        help="Merge type: 'inner' to keep only inferred articles, 'left' to keep all clean articles.",
    )
    return parser.parse_args()


def merge_articles_sentiment(
    articles_path: str | Path,
    sentiment_path: str | Path,
    output_path: str | Path,
    how: str = "inner",
) -> None:
    """Merge article texts with sentiment scores on url and trading_date."""
    logger.info("Reading articles from %s", articles_path)
    articles_df = read_parquet_table(articles_path)

    logger.info("Reading sentiment scores from %s", sentiment_path)
    sentiment_df = read_parquet_table(sentiment_path)

    logger.info("Merging datasets using how='%s' on ['url', 'trading_date']", how)
    merged_df = pd.merge(
        articles_df,
        sentiment_df,
        on=["url", "trading_date"],
        how=how,
    )

    logger.info(
        "Successfully merged: Articles=%d, Sentiment=%d -> Merged=%d rows",
        len(articles_df),
        len(sentiment_df),
        len(merged_df),
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(out_path, index=False)
    logger.info("Saved merged data to %s", out_path)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    merge_articles_sentiment(
        articles_path=args.articles_file,
        sentiment_path=args.sentiment_file,
        output_path=args.output_file,
        how=args.how,
    )


if __name__ == "__main__":
    main()
