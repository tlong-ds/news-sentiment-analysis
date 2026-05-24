import pandas as pd

from src.preprocessing.news_alignment import aggregate_daily_news, align_articles_to_trading_day


def test_align_articles_to_trading_day_respects_market_close_cutoff():
    news = pd.DataFrame(
        {
            "title": ["pre-close", "after-close"],
            "published_at": ["2024-01-08 14:30:00", "2024-01-08 15:00:00"],
            "date": ["2024-01-08", "2024-01-08"],
            "category": ["Chung khoan", "Chung khoan"],
            "body_clean": ["a" * 120, "b" * 140],
        }
    )
    trading_dates = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"])

    aligned = align_articles_to_trading_day(news, trading_dates)

    assert aligned.loc[0, "trading_date"] == pd.Timestamp("2024-01-08")
    assert aligned.loc[0, "alignment_reason"] == "same_session"
    assert aligned.loc[1, "trading_date"] == pd.Timestamp("2024-01-09")
    assert aligned.loc[1, "alignment_reason"] == "after_close_forward"


def test_align_articles_to_trading_day_handles_long_holiday_gaps_without_dropping_rows():
    news = pd.DataFrame(
        {
            "title": ["tet-news"],
            "published_at": ["2024-02-10 09:00:00"],
            "date": ["2024-02-10"],
            "category": ["Vi mo"],
            "body_clean": ["holiday article"],
        }
    )
    trading_dates = pd.to_datetime(["2024-02-08", "2024-02-19", "2024-02-20"])

    aligned = align_articles_to_trading_day(news, trading_dates)

    assert aligned.loc[0, "trading_date"] == pd.Timestamp("2024-02-19")
    assert aligned.loc[0, "alignment_reason"] == "non_trading_forward"
    assert aligned.loc[0, "calendar_gap_days"] == 9


def test_aggregate_daily_news_exposes_bunching_diagnostics():
    aligned = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2024-02-19", "2024-02-19", "2024-02-20"]),
            "category": ["Vi mo", "Chung khoan", "Vi mo"],
            "body_clean": ["a" * 100, "b" * 120, "c" * 110],
            "is_after_close": [0, 1, 0],
            "alignment_reason": ["non_trading_forward", "after_close_forward", "same_session"],
            "calendar_gap_days": [9, 1, 0],
        }
    )

    daily = aggregate_daily_news(aligned)

    assert list(daily.columns) == [
        "date",
        "n_articles",
        "n_categories",
        "mean_body_len",
        "after_close_share",
        "non_trading_share",
        "max_calendar_gap_days",
    ]
    assert daily.loc[0, "n_articles"] == 2
    assert daily.loc[0, "after_close_share"] == 0.5
    assert daily.loc[0, "non_trading_share"] == 0.5
    assert daily.loc[0, "max_calendar_gap_days"] == 9


# ---------------------------------------------------------------------------
# New tests for preprocessing pipeline integration
# ---------------------------------------------------------------------------


def test_published_at_preserved_into_articles_output():
    """Rows with published_at set have has_timestamp=1 and use session alignment."""
    news = pd.DataFrame(
        {
            "title": ["morning post"],
            "published_at": ["2024-03-04 09:30:00"],
            "date": ["2024-03-04"],
            "category": ["Chứng khoán"],
            "body_clean": ["x" * 200],
        }
    )
    trading_dates = pd.to_datetime(["2024-03-04", "2024-03-05"])

    aligned = align_articles_to_trading_day(news, trading_dates)

    assert int(aligned.loc[0, "has_timestamp"]) == 1
    assert aligned.loc[0, "trading_date"] == pd.Timestamp("2024-03-04")
    assert aligned.loc[0, "alignment_reason"] == "same_session"


def test_null_published_at_falls_back_to_date_only_alignment():
    """Rows without published_at (empty string or NaN) use date-only alignment."""
    news = pd.DataFrame(
        {
            "title": ["weekend article"],
            "published_at": [""],  # empty — simulates old scraper schema
            "date": ["2024-03-02"],  # Saturday
            "category": ["Vĩ mô"],
            "body_clean": ["y" * 150],
        }
    )
    trading_dates = pd.to_datetime(["2024-03-01", "2024-03-04"])

    aligned = align_articles_to_trading_day(news, trading_dates)

    # Saturday → shifted to Monday.
    assert int(aligned.loc[0, "has_timestamp"]) == 0
    assert aligned.loc[0, "trading_date"] == pd.Timestamp("2024-03-04")
    assert aligned.loc[0, "alignment_reason"] == "date_only_forward"


def test_after_close_article_shifts_to_next_session():
    """An article published after HoSE close on a trading day shifts forward."""
    news = pd.DataFrame(
        {
            "title": ["late news"],
            "published_at": ["2024-01-10 15:30:00"],  # after 14:45
            "date": ["2024-01-10"],
            "category": ["Doanh nghiệp"],
            "body_clean": ["z" * 180],
        }
    )
    trading_dates = pd.to_datetime(["2024-01-10", "2024-01-11"])

    aligned = align_articles_to_trading_day(news, trading_dates)

    assert aligned.loc[0, "trading_date"] == pd.Timestamp("2024-01-11")
    assert aligned.loc[0, "alignment_reason"] == "after_close_forward"
    assert int(aligned.loc[0, "is_after_close"]) == 1


def test_long_holiday_shift_beyond_7_days_is_not_dropped():
    """Tết-like holiday closures spanning > 7 days are mapped correctly, not dropped."""
    news = pd.DataFrame(
        {
            "title": ["tet-holiday news"],
            "published_at": ["2024-02-12 10:00:00"],
            "date": ["2024-02-12"],
            "category": ["Kinh tế"],
            "body_clean": ["a" * 300],
        }
    )
    # No trading day between Feb 9 and Feb 21 — gap of 12 days.
    trading_dates = pd.to_datetime(["2024-02-08", "2024-02-21", "2024-02-22"])

    aligned = align_articles_to_trading_day(news, trading_dates)

    assert not aligned["trading_date"].isna().any(), "Long-holiday row must not be dropped."
    assert aligned.loc[0, "trading_date"] == pd.Timestamp("2024-02-21")
    assert aligned.loc[0, "calendar_gap_days"] == 9
