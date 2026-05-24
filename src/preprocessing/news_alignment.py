"""Trading-session alignment helpers for Vietnamese news pipelines.

The original notebook aligns articles to the next valid trading day only at
the calendar-date level. This module adds two things that matter for a
volatility-forecasting study:

1. A HoSE close cutoff so same-day after-close articles are shifted forward.
2. Alignment diagnostics so long holiday bunching can be measured rather than
   silently absorbed into daily article counts.
"""

from __future__ import annotations

from datetime import time

import pandas as pd

DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"
DEFAULT_MARKET_CLOSE = time(14, 45)


def _coerce_market_timestamp(series: pd.Series, timezone: str) -> pd.Series:
    """Parse timestamps and interpret naive values in local market time."""
    parsed = pd.to_datetime(series, errors="coerce")
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_localize(timezone, nonexistent="NaT", ambiguous="NaT")
    return parsed.dt.tz_convert(timezone)


def _next_trading_day(
    origin_day: pd.Timestamp,
    trading_index: pd.DatetimeIndex,
    *,
    include_same_day: bool,
) -> pd.Timestamp | pd.NaT:
    if pd.isna(origin_day):
        return pd.NaT

    pos = trading_index.searchsorted(origin_day)
    if include_same_day and pos < len(trading_index) and trading_index[pos] == origin_day:
        return trading_index[pos]
    if not include_same_day and pos < len(trading_index) and trading_index[pos] == origin_day:
        pos += 1
    if pos >= len(trading_index):
        return pd.NaT
    return trading_index[pos]


def align_articles_to_trading_day(
    news_df: pd.DataFrame,
    trading_dates: pd.Series | pd.DatetimeIndex | list[str],
    *,
    timestamp_col: str = "published_at",
    date_col: str = "date",
    category_col: str = "category",
    market_close: time = DEFAULT_MARKET_CLOSE,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Align articles to the trading session that could have absorbed them.

    Output columns added:
    - ``origin_date``: article's calendar date in local market time
    - ``trading_date``: aligned trading session
    - ``has_timestamp``: whether intraday cutoff logic was available
    - ``is_after_close``: article arrived after the session close
    - ``calendar_gap_days``: calendar days between origin date and trading date
    - ``alignment_reason``: one of same_session/date_only_same_day/
      after_close_forward/date_only_forward/non_trading_forward/unmapped
    """
    if date_col not in news_df.columns and timestamp_col not in news_df.columns:
        raise ValueError(f"Expected at least one of {date_col!r} or {timestamp_col!r}.")

    aligned = news_df.copy()
    trading_index = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize().sort_values().unique()
    if trading_index.empty:
        raise ValueError("trading_dates cannot be empty.")

    published_at = (
        _coerce_market_timestamp(aligned[timestamp_col], timezone)
        if timestamp_col in aligned.columns
        else pd.Series(pd.NaT, index=aligned.index, dtype=f"datetime64[ns, {timezone}]")
    )
    calendar_dates = (
        pd.to_datetime(aligned[date_col], errors="coerce").dt.normalize()
        if date_col in aligned.columns
        else pd.Series(pd.NaT, index=aligned.index)
    )

    has_timestamp = published_at.notna()
    origin_dates = pd.Series(calendar_dates, index=aligned.index)
    origin_dates.loc[has_timestamp] = published_at.loc[has_timestamp].dt.tz_localize(None).dt.normalize()
    is_trading_origin = origin_dates.isin(trading_index)
    is_after_close = pd.Series(False, index=aligned.index)
    if has_timestamp.any():
        is_after_close.loc[has_timestamp] = (
            published_at.loc[has_timestamp].dt.time > market_close
        ) & is_trading_origin.loc[has_timestamp]

    trading_days: list[pd.Timestamp | pd.NaT] = []
    reasons: list[str] = []
    for idx in aligned.index:
        origin_day = origin_dates.loc[idx]
        has_time = bool(has_timestamp.loc[idx])
        trading_origin = bool(is_trading_origin.loc[idx])
        after_close = bool(is_after_close.loc[idx])

        if pd.isna(origin_day):
            trading_days.append(pd.NaT)
            reasons.append("unmapped")
            continue

        if trading_origin and not after_close:
            trading_days.append(_next_trading_day(origin_day, trading_index, include_same_day=True))
            reasons.append("same_session" if has_time else "date_only_same_day")
            continue

        if trading_origin and after_close:
            trading_days.append(_next_trading_day(origin_day, trading_index, include_same_day=False))
            reasons.append("after_close_forward")
            continue

        trading_days.append(_next_trading_day(origin_day, trading_index, include_same_day=True))
        reasons.append("non_trading_forward" if has_time else "date_only_forward")

    aligned["origin_date"] = origin_dates
    aligned["trading_date"] = pd.to_datetime(trading_days)
    aligned["has_timestamp"] = has_timestamp.astype(int)
    aligned["is_after_close"] = is_after_close.astype(int)
    aligned["alignment_reason"] = reasons
    aligned["calendar_gap_days"] = (
        aligned["trading_date"] - aligned["origin_date"]
    ).dt.days

    if category_col in aligned.columns:
        aligned["category"] = aligned[category_col]

    return aligned


def aggregate_daily_news(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate aligned article rows into daily controls with bunching signals."""
    if "trading_date" not in aligned_df.columns:
        raise ValueError("aligned_df must contain 'trading_date'.")

    aggregated = aligned_df.copy()
    if "body_len" not in aggregated.columns:
        text_col = "body_clean" if "body_clean" in aggregated.columns else "body"
        if text_col not in aggregated.columns:
            raise ValueError("Need one of 'body_len', 'body_clean', or 'body'.")
        aggregated["body_len"] = aggregated[text_col].fillna("").astype(str).str.len()

    aggregated["is_non_trading_origin"] = aggregated["alignment_reason"].isin(
        {"non_trading_forward", "date_only_forward"}
    ).astype(int)

    daily = (
        aggregated.dropna(subset=["trading_date"])
        .groupby("trading_date", as_index=False)
        .agg(
            n_articles=("trading_date", "size"),
            n_categories=("category", "nunique"),
            mean_body_len=("body_len", "mean"),
            after_close_share=("is_after_close", "mean"),
            non_trading_share=("is_non_trading_origin", "mean"),
            max_calendar_gap_days=("calendar_gap_days", "max"),
        )
        .rename(columns={"trading_date": "date"})
    )

    return daily
