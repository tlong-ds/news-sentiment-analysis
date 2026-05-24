"""Regression tests for the modular preprocessing pipeline entrypoint.

Uses small synthetic fixtures so the tests run fast without touching the
~966 MB production CafeF CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import (
    ARTICLES_CLEAN_COLUMNS,
    DAILY_NEWS_PRICES_COLUMNS,
    build_preprocessed_outputs,
    export_preprocessed_outputs,
)


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_news(tmp_path: Path) -> Path:
    """Write a minimal CafeF-style CSV with 20 synthetic articles."""
    rows = []
    trading_days = pd.date_range("2024-01-02", periods=10, freq="B")  # 10 business days
    for i, dt in enumerate(trading_days):
        # 2 articles per trading day, one pre-close, one post-close.
        rows.append(
            {
                "url": f"https://cafef.vn/test-{i:02d}a.chn",
                "source": "cafef",
                "category": "Chứng khoán",
                "title": f"Morning article {i}",
                "date": dt.strftime("%Y-%m-%d"),
                "published_at": dt.strftime("%Y-%m-%d") + " 09:00:00",
                "body": "x " * 200,  # 400 chars — above 100-char threshold
            }
        )
        rows.append(
            {
                "url": f"https://cafef.vn/test-{i:02d}b.chn",
                "source": "cafef",
                "category": "Doanh nghiệp",
                "title": f"Evening article {i}",
                "date": dt.strftime("%Y-%m-%d"),
                "published_at": dt.strftime("%Y-%m-%d") + " 15:30:00",  # after close
                "body": "y " * 200,
            }
        )
    # One short article (should be filtered).
    rows.append(
        {
            "url": "https://cafef.vn/short.chn",
            "source": "cafef",
            "category": "Vĩ mô",
            "title": "Stub",
            "date": "2024-01-02",
            "published_at": "2024-01-02 10:00:00",
            "body": "short",  # 5 chars — below threshold
        }
    )
    # One article with no published_at (old schema row).
    rows.append(
        {
            "url": "https://cafef.vn/notime.chn",
            "source": "cafef",
            "category": "Tài chính ngân hàng",
            "title": "No timestamp article",
            "date": "2024-01-02",
            "published_at": "",
            "body": "z " * 200,
        }
    )
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "news_VN_cafef.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def _make_synthetic_prices(tmp_path: Path) -> Path:
    """Write a minimal LSEG-style price CSV covering 10 business days."""
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    np.random.seed(42)
    close = 1200.0 + np.cumsum(np.random.normal(0, 5, len(dates)))
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "TRDPRC_1": close,
            "OPEN_PRC": close - np.random.uniform(0, 3, len(dates)),
            "HIGH_1": close + np.random.uniform(0, 5, len(dates)),
            "LOW_1": close - np.random.uniform(0, 5, len(dates)),
            "ACVOL_UNS": np.random.randint(50_000_000, 200_000_000, len(dates)),
        }
    )
    csv_path = tmp_path / "prices_VN.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_preprocessed_outputs_end_to_end(tmp_path: Path):
    """build_preprocessed_outputs returns correct schemas and plausible counts."""
    news_path = _make_synthetic_news(tmp_path)
    prices_path = _make_synthetic_prices(tmp_path)

    articles_df, daily_df, diagnostics = build_preprocessed_outputs(
        news_path,
        prices_path,
        min_body_len=100,
    )

    # ---- articles_df schema ------------------------------------------------
    missing_cols = [c for c in ARTICLES_CLEAN_COLUMNS if c not in articles_df.columns]
    assert not missing_cols, f"articles_df missing columns: {missing_cols}"

    # Short article (5 chars) should be filtered out.
    assert len(articles_df) < 22, "Short-article filter should remove at least 1 row."

    # After-close rows should be shifted to the next trading day.
    after_close = articles_df[articles_df["alignment_reason"] == "after_close_forward"]
    assert len(after_close) > 0, "Expected at least some after-close forward shifts."

    # Row with no published_at should have has_timestamp=0.
    no_ts = articles_df[articles_df["url"].str.contains("notime")]
    assert len(no_ts) == 1
    assert int(no_ts.iloc[0]["has_timestamp"]) == 0
    assert no_ts.iloc[0]["alignment_reason"] in {"date_only_same_day", "date_only_forward"}

    # Row with published_at should have has_timestamp=1.
    with_ts = articles_df[articles_df["url"].str.contains("test-00a")]
    assert len(with_ts) == 1
    assert int(with_ts.iloc[0]["has_timestamp"]) == 1

    # ---- daily_df schema ---------------------------------------------------
    missing_d = [c for c in DAILY_NEWS_PRICES_COLUMNS if c not in daily_df.columns]
    assert not missing_d, f"daily_df missing columns: {missing_d}"

    # Every trading day in the price file should appear in daily_df.
    assert len(daily_df) == 10, (
        f"Expected 10 daily rows (one per trading day); got {len(daily_df)}"
    )

    # Zero-news days should have n_articles == 0 (not NaN).
    assert daily_df["n_articles"].isna().sum() == 0

    # ---- diagnostics -------------------------------------------------------
    required_keys = {
        "raw_cafef_row_count",
        "cleaned_article_row_count",
        "processed_daily_row_count",
        "published_at_non_null_share",
        "timestamp_based_alignment_share",
        "date_only_fallback_share",
        "after_close_forward_shifts",
        "non_trading_day_forward_shifts",
        "daily_vs_price_explanation",
    }
    missing_keys = required_keys - set(diagnostics.keys())
    assert not missing_keys, f"diagnostics missing keys: {missing_keys}"

    # published_at is present for all but 1 article in synthetic data.
    assert diagnostics["published_at_non_null_share"] > 0.0

    # Short article was removed.
    assert diagnostics["short_articles_removed"] >= 1


def test_export_preprocessed_outputs_writes_correct_files(tmp_path: Path):
    """export_preprocessed_outputs writes 3 files with correct schemas."""
    news_path = _make_synthetic_news(tmp_path)
    prices_path = _make_synthetic_prices(tmp_path)

    articles_df, daily_df, diagnostics = build_preprocessed_outputs(
        news_path,
        prices_path,
        min_body_len=100,
    )

    out_dir = tmp_path / "processed"
    backup_path = tmp_path / "news_VN_cafef_backup_20240101.csv"
    backup_path.touch()  # Simulate backup existence.

    paths = export_preprocessed_outputs(
        articles_df,
        daily_df,
        diagnostics,
        out_dir=out_dir,
        backup_path=backup_path,
    )

    # All 3 output files must exist.
    assert paths["articles_clean"].exists()
    assert paths["daily_news_prices"].exists()
    assert paths["diagnostics"].exists()

    # Reload and verify column contracts.
    articles_reload = pd.read_csv(paths["articles_clean"])
    daily_reload = pd.read_csv(paths["daily_news_prices"])

    assert set(ARTICLES_CLEAN_COLUMNS).issubset(set(articles_reload.columns))
    assert set(DAILY_NEWS_PRICES_COLUMNS).issubset(set(daily_reload.columns))

    # Diagnostics JSON should record the backup path.
    import json
    diag = json.loads(paths["diagnostics"].read_text(encoding="utf-8"))
    assert "backup_file_path" in diag
    assert str(backup_path.resolve()) in diag["backup_file_path"]

    # daily_news_prices must preserve one row per trading day.
    assert len(daily_reload) == 10

    # n_articles must not have NaN (zero-filled).
    assert daily_reload["n_articles"].isna().sum() == 0


def test_daily_export_preserves_zero_news_trading_days(tmp_path: Path):
    """Zero-news trading days stay in daily_df with n_articles == 0."""
    # Only 1 article on the first day; remaining 9 days have no news.
    rows = [
        {
            "url": "https://cafef.vn/one.chn",
            "source": "cafef",
            "category": "Chứng khoán",
            "title": "Only article",
            "date": "2024-01-02",
            "published_at": "2024-01-02 09:00:00",
            "body": "a " * 200,
        }
    ]
    df = pd.DataFrame(rows)
    news_path = tmp_path / "news_sparse.csv"
    df.to_csv(news_path, index=False)

    prices_path = _make_synthetic_prices(tmp_path)
    _, daily_df, _ = build_preprocessed_outputs(news_path, prices_path, min_body_len=100)

    # All 10 trading days must be present.
    assert len(daily_df) == 10
    # 9 days should have zero articles.
    zero_days = (daily_df["n_articles"] == 0).sum()
    assert zero_days == 9, f"Expected 9 zero-news days; got {zero_days}"
    # n_articles must not contain NaN.
    assert daily_df["n_articles"].isna().sum() == 0
