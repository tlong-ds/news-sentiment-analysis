"""Modular preprocessing pipeline for CafeF news and VN-Index price data.

This module is the single authoritative entrypoint for the pre-sentiment data
workflow.  It owns:

- Loading raw news (handles both old schema without ``published_at`` and the
  new schema that includes it).
- Article body cleaning and short-article filtering.
- Trading-day alignment via :func:`align_articles_to_trading_day`.
- Daily aggregation via :func:`aggregate_daily_news`.
- Merge to the full trading-day price calendar (zero-fills zero-news days).
- Export of processed article and daily outputs.
- Machine-readable diagnostics summary.

Public API
----------
- :func:`build_preprocessed_outputs` — main builder, returns DataFrames + diagnostics.
- :func:`export_preprocessed_outputs` — writes CSVs and the diagnostics JSON.

CLI usage::

    python -m src.preprocessing.pipeline \\
        --raw-news data/raw/news_VN_cafef.csv \\
        --prices data/raw/prices_VN.csv \\
        --out-dir data/processed

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.news_alignment import (
    aggregate_daily_news,
    align_articles_to_trading_day,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field size limit for large article bodies
# ---------------------------------------------------------------------------
csv.field_size_limit(2_147_483_647)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Columns produced in articles_clean.csv.
ARTICLES_CLEAN_COLUMNS: list[str] = [
    "url",
    "source",
    "category",
    "title",
    "date",
    "published_at",
    "origin_date",
    "trading_date",
    "has_timestamp",
    "is_after_close",
    "alignment_reason",
    "calendar_gap_days",
    "body_clean",
    "body_len",
]

#: Columns produced in daily_news_prices.csv.
DAILY_NEWS_PRICES_COLUMNS: list[str] = [
    "date",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "log_return",
    "n_articles",
    "n_categories",
    "mean_body_len",
    "after_close_share",
    "non_trading_share",
    "max_calendar_gap_days",
]

#: Pattern for HTML tags.
_RE_HTML = re.compile(r"<[^>]+>")
#: Pattern for URLs.
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
#: Pattern for email addresses.
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
#: Pattern for runs of whitespace.
_RE_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_body(text: str) -> str:
    """Strip HTML, URLs, emails, and normalize whitespace.

    Args:
        text: Raw article body string.

    Returns:
        Cleaned body string.
    """
    if not text or not isinstance(text, str):
        return ""
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------

def _load_prices(price_path: Path) -> pd.DataFrame:
    """Load VN-Index price CSV and compute ``log_return``.

    Args:
        price_path: Path to ``prices_VN.csv`` (LSEG-style columns).

    Returns:
        DataFrame with columns ``date, close, open, high, low, volume,
        log_return``, sorted ascending by ``date``.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(price_path)
    rename = {
        "Date": "date",
        "TRDPRC_1": "close",
        "OPEN_PRC": "open",
        "HIGH_1": "high",
        "LOW_1": "low",
        "ACVOL_UNS": "volume",
    }
    df = df.rename(columns=rename)
    required = {"date", "close", "open", "high", "low", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price CSV missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df[["date", "close", "open", "high", "low", "volume", "log_return"]]


# ---------------------------------------------------------------------------
# Raw news loading
# ---------------------------------------------------------------------------

def _load_raw_news(raw_news_path: Path) -> pd.DataFrame:
    """Load raw CafeF news CSV, tolerating a missing ``published_at`` column.

    Old scraper runs produced ``url,source,category,title,date,body``.
    New runs add ``published_at`` between ``date`` and ``body``.  Both
    layouts are handled: when ``published_at`` is absent the column is
    synthesised as an empty string so downstream alignment logic can use
    date-only fallback uniformly.

    Args:
        raw_news_path: Path to ``news_VN_cafef.csv``.

    Returns:
        DataFrame with at least ``url, source, category, title, date,
        published_at, body`` columns.
    """
    logger.info("Loading raw news: %s", raw_news_path)
    df = pd.read_csv(
        raw_news_path,
        dtype=str,
        keep_default_na=False,
        engine="python",
    )
    if "published_at" not in df.columns:
        logger.info(
            "Raw CSV has no 'published_at' column — adding empty column for "
            "date-only alignment fallback."
        )
        df["published_at"] = ""
    logger.info("Raw news loaded: %d rows, columns: %s", len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_preprocessed_outputs(
    raw_news_path: str | Path,
    price_path: str | Path,
    *,
    min_body_len: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build processed article and daily frames plus diagnostics.

    Steps
    -----
    1. Load raw news; synthesise ``published_at`` if absent.
    2. Clean body text; filter short articles (``body_len < min_body_len``).
    3. Align articles to trading day using :func:`align_articles_to_trading_day`.
    4. Aggregate to daily news controls via :func:`aggregate_daily_news`.
    5. Merge with full price trading-day calendar; zero-fill zero-news days.
    6. Build diagnostics summary.

    Args:
        raw_news_path: Path to raw CafeF news CSV.
        price_path: Path to VN-Index price CSV.
        min_body_len: Minimum ``body_len`` (characters) to keep an article.

    Returns:
        Tuple of:
        - ``articles_df``: Article-level DataFrame with alignment diagnostics.
        - ``daily_df``: Daily trading-day DataFrame merged with price data.
        - ``diagnostics``: Machine-readable summary dict.
    """
    raw_news_path = Path(raw_news_path)
    price_path = Path(price_path)

    # ---- Step 1: load raw news -----------------------------------------------
    news_raw = _load_raw_news(raw_news_path)
    raw_row_count = len(news_raw)

    # ---- Step 2: clean body text ---------------------------------------------
    logger.info("Cleaning article body text …")
    news_raw["body_clean"] = news_raw["body"].apply(_clean_body)
    news_raw["body_len"] = news_raw["body_clean"].str.len()

    before_filter = len(news_raw)
    news_clean = news_raw[news_raw["body_len"] >= min_body_len].copy()
    logger.info(
        "Short-article filter (<%d chars): %d removed, %d kept",
        min_body_len,
        before_filter - len(news_clean),
        len(news_clean),
    )

    # ---- Step 3: load prices and align articles ------------------------------
    prices = _load_prices(price_path)
    trading_dates = prices["date"]

    logger.info("Aligning %d articles to trading days …", len(news_clean))
    # Use published_at only when non-empty; date column always present.
    # news_alignment handles empty published_at via NaT coercion.
    aligned = align_articles_to_trading_day(
        news_clean,
        trading_dates,
        timestamp_col="published_at",
        date_col="date",
        category_col="category",
    )

    # Drop rows that could not be mapped to any trading day.
    unmapped_count = aligned["trading_date"].isna().sum()
    if unmapped_count > 0:
        logger.warning(
            "%d articles could not be mapped to a trading day (unmapped) — dropping.",
            unmapped_count,
        )
    aligned = aligned.dropna(subset=["trading_date"])

    # Normalise trading_date type for consistent CSV serialisation.
    aligned["trading_date"] = pd.to_datetime(aligned["trading_date"]).dt.date

    # Build articles_df with the contracted column set.
    articles_df = aligned.reindex(
        columns=[c for c in ARTICLES_CLEAN_COLUMNS if c in aligned.columns]
    ).copy()
    # Ensure all contract columns exist (fill with empty if somehow absent).
    for col in ARTICLES_CLEAN_COLUMNS:
        if col not in articles_df.columns:
            articles_df[col] = ""

    articles_df = articles_df[ARTICLES_CLEAN_COLUMNS]

    # ---- Step 4: aggregate daily news ----------------------------------------
    logger.info("Aggregating daily news controls …")
    # aggregate_daily_news needs trading_date as Timestamp.
    aligned_for_agg = aligned.copy()
    aligned_for_agg["trading_date"] = pd.to_datetime(aligned_for_agg["trading_date"])
    daily_news = aggregate_daily_news(aligned_for_agg)
    daily_news["date"] = pd.to_datetime(daily_news["date"])

    # ---- Step 5: merge with full trading-day price calendar ------------------
    logger.info("Merging daily news with price calendar …")
    daily_df = prices.merge(daily_news, on="date", how="left")

    # Zero-fill news controls on zero-news trading days.
    zero_fill: dict[str, int | float] = {
        "n_articles": 0,
        "n_categories": 0,
        "mean_body_len": 0.0,
        "after_close_share": 0.0,
        "non_trading_share": 0.0,
        "max_calendar_gap_days": 0,
    }
    for col, default in zero_fill.items():
        if col in daily_df.columns:
            daily_df[col] = daily_df[col].fillna(default)
        else:
            daily_df[col] = default

    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    # Keep only contracted columns that are present.
    daily_df = daily_df.reindex(
        columns=[c for c in DAILY_NEWS_PRICES_COLUMNS if c in daily_df.columns]
    )

    # ---- Step 6: diagnostics -------------------------------------------------
    has_timestamp_series = aligned["has_timestamp"].astype(bool)
    timestamp_share = float(has_timestamp_series.mean()) if len(aligned) > 0 else 0.0
    date_only_share = 1.0 - timestamp_share

    after_close_count = int(
        aligned["alignment_reason"].eq("after_close_forward").sum()
    )
    non_trading_count = int(
        aligned["alignment_reason"].isin(
            {"non_trading_forward", "date_only_forward"}
        ).sum()
    )

    price_row_count = len(prices)
    daily_row_count = len(daily_df)
    row_diff = price_row_count - daily_row_count
    row_diff_explanation = (
        "daily_news_prices row count equals price row count (all trading days preserved)."
        if row_diff == 0
        else (
            f"Price CSV has {price_row_count} rows; daily output has {daily_row_count} rows "
            f"(difference: {row_diff}). This can occur if some price dates fall entirely "
            "outside the news article date range."
        )
    )

    diagnostics: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_news_path": str(raw_news_path.resolve()),
        "price_path": str(price_path.resolve()),
        "raw_cafef_row_count": raw_row_count,
        "after_short_filter_row_count": len(news_clean),
        "short_articles_removed": before_filter - len(news_clean),
        "min_body_len_threshold": min_body_len,
        "unmapped_articles_dropped": int(unmapped_count),
        "cleaned_article_row_count": len(articles_df),
        "published_at_non_null_share": timestamp_share,
        "timestamp_based_alignment_share": timestamp_share,
        "date_only_fallback_share": date_only_share,
        "after_close_forward_shifts": after_close_count,
        "non_trading_day_forward_shifts": non_trading_count,
        "processed_daily_row_count": daily_row_count,
        "price_row_count": price_row_count,
        "daily_vs_price_row_diff": row_diff,
        "daily_vs_price_explanation": row_diff_explanation,
    }

    logger.info(
        "Preprocessing complete — %d articles, %d daily rows.",
        len(articles_df),
        len(daily_df),
    )
    return articles_df, daily_df, diagnostics


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def export_preprocessed_outputs(
    articles_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    diagnostics: dict[str, Any],
    *,
    out_dir: str | Path = "data/processed",
    backup_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write processed outputs and diagnostics to disk.

    Args:
        articles_df: Article-level DataFrame from :func:`build_preprocessed_outputs`.
        daily_df: Daily DataFrame from :func:`build_preprocessed_outputs`.
        diagnostics: Diagnostics dict from :func:`build_preprocessed_outputs`.
        out_dir: Output directory (created if absent).
        backup_path: Optional path of the raw backup file to record in diagnostics.

    Returns:
        Dict mapping output names to their absolute paths:
        ``{"articles_clean": ..., "daily_news_prices": ..., "diagnostics": ...}``
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if backup_path is not None:
        diagnostics = {**diagnostics, "backup_file_path": str(Path(backup_path).resolve())}

    articles_path = out / "articles_clean.csv"
    daily_path = out / "daily_news_prices.csv"
    diagnostics_path = out / "preprocessing_diagnostics.json"

    logger.info("Writing articles_clean.csv (%d rows) …", len(articles_df))
    articles_df.to_csv(articles_path, index=False, encoding="utf-8")

    logger.info("Writing daily_news_prices.csv (%d rows) …", len(daily_df))
    daily_df.to_csv(daily_path, index=False, encoding="utf-8")

    logger.info("Writing preprocessing_diagnostics.json …")
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    paths = {
        "articles_clean": articles_path.resolve(),
        "daily_news_prices": daily_path.resolve(),
        "diagnostics": diagnostics_path.resolve(),
    }
    for name, path in paths.items():
        logger.info("  %s → %s", name, path)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the modular CafeF preprocessing pipeline."
    )
    parser.add_argument(
        "--raw-news",
        default="data/raw/news_VN_cafef.csv",
        help="Path to raw CafeF news CSV (default: data/raw/news_VN_cafef.csv).",
    )
    parser.add_argument(
        "--prices",
        default="data/raw/prices_VN.csv",
        help="Path to VN-Index price CSV (default: data/raw/prices_VN.csv).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Output directory for processed files (default: data/processed).",
    )
    parser.add_argument(
        "--min-body-len",
        type=int,
        default=100,
        help="Minimum article body length in characters (default: 100).",
    )
    parser.add_argument(
        "--backup-path",
        default=None,
        help="Path of the raw backup file to record in diagnostics (optional).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    articles_df, daily_df, diagnostics = build_preprocessed_outputs(
        args.raw_news,
        args.prices,
        min_body_len=args.min_body_len,
    )

    paths = export_preprocessed_outputs(
        articles_df,
        daily_df,
        diagnostics,
        out_dir=args.out_dir,
        backup_path=args.backup_path,
    )

    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
