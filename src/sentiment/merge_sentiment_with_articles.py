"""Merge cleaned article records with inferred sentiment scores."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import INTERIM_DATA_DIR, SENTIMENT_DATA_DIR
from src.utils.io import read_parquet_table

logger = logging.getLogger(__name__)

JOIN_KEYS = ["url", "trading_date"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge cleaned article records with inferred sentiment scores."
    )
    parser.add_argument(
        "--articles-file",
        default=f"{INTERIM_DATA_DIR}/articles_clean.parquet",
        help="Path to the cleaned article parquet file.",
    )
    parser.add_argument(
        "--sentiment-file",
        default=f"{SENTIMENT_DATA_DIR}/article_sentiment_scores.parquet",
        help="Path to the inferred sentiment scores parquet file.",
    )
    parser.add_argument(
        "--output-file",
        default=f"{INTERIM_DATA_DIR}/articles_with_sentiment.parquet",
        help="Path to save the merged parquet output.",
    )
    parser.add_argument(
        "--how",
        choices=["inner", "left"],
        default="left",
        help=(
            "Merge type: 'inner' keeps only articles with sentiment scores; "
            "'left' keeps all cleaned articles and leaves missing sentiment as nulls."
        ),
    )
    return parser.parse_args()


def _normalize_join_keys(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(JOIN_KEYS) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required join columns: {missing}")

    out = df.copy()
    out["url"] = out["url"].astype(str).str.strip()
    out["trading_date"] = pd.to_datetime(out["trading_date"], errors="coerce").dt.date
    return out


def merge_articles_sentiment(
    articles_path: str | Path,
    sentiment_path: str | Path,
    output_path: str | Path,
    how: str = "left",
) -> None:
    """Merge article records with sentiment scores on ['url', 'trading_date']."""
    logger.info("Reading articles from %s", articles_path)
    articles_df = _normalize_join_keys(
        read_parquet_table(articles_path), name="articles"
    )

    logger.info("Reading sentiment scores from %s", sentiment_path)
    sentiment_df = _normalize_join_keys(
        read_parquet_table(sentiment_path), name="sentiment"
    )

    articles_dupes = int(articles_df.duplicated(subset=JOIN_KEYS).sum())
    sentiment_dupes = int(sentiment_df.duplicated(subset=JOIN_KEYS).sum())
    if articles_dupes:
        logger.warning(
            "Found %d duplicate article rows on join keys %s; keeping first occurrence.",
            articles_dupes,
            JOIN_KEYS,
        )
        articles_df = articles_df.drop_duplicates(subset=JOIN_KEYS, keep="first")
    if sentiment_dupes:
        logger.warning(
            "Found %d duplicate sentiment rows on join keys %s; keeping last occurrence.",
            sentiment_dupes,
            JOIN_KEYS,
        )
        sentiment_df = sentiment_df.drop_duplicates(subset=JOIN_KEYS, keep="last")

    logger.info("Merging datasets using how='%s' on %s", how, JOIN_KEYS)
    merged_df = articles_df.merge(
        sentiment_df,
        on=JOIN_KEYS,
        how=how,
        indicator=True,
        suffixes=("", "_sentiment"),
    )

    key_outer = (
        articles_df[JOIN_KEYS]
        .drop_duplicates()
        .merge(
            sentiment_df[JOIN_KEYS].drop_duplicates(),
            on=JOIN_KEYS,
            how="outer",
            indicator=True,
        )
    )

    report = {
        "articles_rows": int(len(articles_df)),
        "sentiment_rows": int(len(sentiment_df)),
        "merged_rows": int(len(merged_df)),
        "articles_duplicate_keys": articles_dupes,
        "sentiment_duplicate_keys": sentiment_dupes,
        "key_match": {
            "both": int((key_outer["_merge"] == "both").sum()),
            "articles_only": int((key_outer["_merge"] == "left_only").sum()),
            "sentiment_only": int((key_outer["_merge"] == "right_only").sum()),
        },
        "merge_counts": merged_df["_merge"].value_counts().to_dict(),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.drop(columns=["_merge"], errors="ignore").to_parquet(
        out_path, index=False
    )
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "Saved merged articles+sentiment -> %s (report: %s)",
        out_path,
        out_path.with_suffix(".report.json"),
    )


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
