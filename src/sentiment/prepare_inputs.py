"""Prepare CafeF sentiment inputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import CAFEF_DATA_DIR, INTERIM_DATA_DIR
from src.sentiment.common import (
    body_lead,
    build_input_text,
    count_tokens,
    normalize_text,
    token_stats,
)
from src.utils.io import read_parquet_table

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CafeF sentiment input tables."
    )
    parser.add_argument(
        "--cafef-input", default=f"{INTERIM_DATA_DIR}/articles_clean.parquet"
    )
    parser.add_argument(
        "--cafef-output", default=f"{CAFEF_DATA_DIR}/cafef_input.parquet"
    )
    parser.add_argument(
        "--report-file", default=f"{CAFEF_DATA_DIR}/input_preparation_report.json"
    )
    parser.add_argument("--max-body-chars", type=int, default=300)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=220)
    return parser.parse_args()


def _prepare_frame(
    df: pd.DataFrame,
    *,
    article_id_col: str,
    source_default: str,
    category_col: str | None,
    date_col: str,
    title_col: str,
    body_col: str,
    include_url: bool = False,
    include_trading_date: bool = False,
    max_body_chars: int = 300,
    min_tokens: int = 5,
    max_tokens: int = 220,
) -> pd.DataFrame:
    prepared = pd.DataFrame(
        {
            "article_id": df[article_id_col].astype(str),
            "source": df["source"].astype(str)
            if "source" in df.columns
            else source_default,
            "category": df[category_col].astype(str)
            if category_col and category_col in df.columns
            else "",
            "date": df[date_col].astype(str),
            "title": df[title_col].fillna("").astype(str),
        }
    )
    prepared["body_lead"] = (
        df[body_col]
        .fillna("")
        .astype(str)
        .map(lambda value: body_lead(value, max_chars=max_body_chars))
    )
    prepared["input_text"] = [
        build_input_text(title, lead)
        for title, lead in zip(prepared["title"], prepared["body_lead"])
    ]
    prepared["token_count"] = (
        prepared["input_text"].map(normalize_text).map(count_tokens)
    )

    if include_url:
        prepared["url"] = df["url"].astype(str)
    if include_trading_date:
        prepared["trading_date"] = df["trading_date"].astype(str)

    prepared = prepared[prepared["token_count"].between(min_tokens, max_tokens)].copy()
    return prepared.reset_index(drop=True)


def build_input_report(
    raw_df: pd.DataFrame, prepared_df: pd.DataFrame, sample_size: int = 1000
) -> dict:
    sample = prepared_df.head(sample_size)
    return {
        "raw_rows": int(len(raw_df)),
        "prepared_rows": int(len(prepared_df)),
        "dropped_rows": int(len(raw_df) - len(prepared_df)),
        "token_stats_full": token_stats(prepared_df["token_count"]),
        "token_stats_sample": token_stats(sample["token_count"]),
        "sample_size_for_p95": int(min(sample_size, len(prepared_df))),
        "p95_under_200": bool(token_stats(sample["token_count"])["p95"] < 200.0)
        if len(sample)
        else False,
    }


def prepare_cafef_inputs(
    articles_clean_path: str | Path,
    output_path: str | Path,
    *,
    max_body_chars: int = 300,
    min_tokens: int = 5,
    max_tokens: int = 220,
) -> pd.DataFrame:
    df = read_parquet_table(articles_clean_path)
    prepared = _prepare_frame(
        df,
        article_id_col="url",
        source_default="cafef",
        category_col="category",
        date_col="date",
        title_col="title",
        body_col="body_clean",
        include_url=True,
        include_trading_date=True,
        max_body_chars=max_body_chars,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    front_cols = [
        "article_id",
        "url",
        "trading_date",
        "source",
        "category",
        "date",
        "title",
        "body_lead",
        "input_text",
        "token_count",
    ]
    prepared = prepared[front_cols]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(output_path, index=False)
    logger.info("Prepared %d CafeF rows -> %s", len(prepared), output_path)
    return prepared


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cafef_raw_df = read_parquet_table(args.cafef_input)
    cafef_prepared = prepare_cafef_inputs(
        args.cafef_input,
        args.cafef_output,
        max_body_chars=args.max_body_chars,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    report = {
        "cafef": build_input_report(cafef_raw_df, cafef_prepared),
        "max_body_chars": args.max_body_chars,
        "token_filter": {"min_tokens": args.min_tokens, "max_tokens": args.max_tokens},
    }
    Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_file).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote input preparation report -> %s", args.report_file)


if __name__ == "__main__":
    main()
