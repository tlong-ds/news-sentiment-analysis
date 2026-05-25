"""vnstock-based news source — ticker-level company news via API.

Uses the ``vnstock`` Python library to fetch dated news articles for
individual stock tickers.  No web scraping required — the library wraps
the broker APIs used by Vietnamese securities firms.

Limitations:
    - ``company.news()`` has no native date-range parameter; we fetch
      whatever the API returns and post-filter by date.
    - Coverage depth varies per ticker and data source.

Install: ``pip install vnstock``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    SOURCE_OUTPUTS,
    VNSTOCK_MAX_PAGES,
    VNSTOCK_PAGE_SIZE,
    VNSTOCK_PROVIDER_ORDER,
    VNSTOCK_SYMBOLS,
)
from src.ingestion.base import Article, SourceStats, append_articles
from src.utils.date_utils import parse_iso_date

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ArticleMetrics:
    rows_with_real_urls: int = 0
    rows_with_published_at: int = 0
    title_body_fallbacks: int = 0


def _normalize_text(value: Any) -> str:
    """Convert scalar values to a clean string without leaking literal nulls."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_date_fields(raw_value: str) -> tuple[str, str]:
    """Return (date, published_at) from a raw vnstock date field.

    `published_at` is only filled when the upstream value appears to include
    intraday time. Date-only strings keep `published_at` empty to avoid
    inventing midnight precision.
    """
    parsed = pd.Timestamp(raw_value)
    iso_date = parsed.strftime("%Y-%m-%d")
    has_intraday_time = any(token in raw_value for token in ("T", ":")) or any(
        [parsed.hour, parsed.minute, parsed.second, parsed.microsecond]
    )
    published_at = parsed.isoformat() if has_intraday_time else ""
    return iso_date, published_at


def _clean_provider_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column labels while preserving provider-emitted names."""
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    return cleaned


def _make_company(symbol: str, provider: str):
    """Build a vnstock Company adapter for a provider."""
    from vnstock.api.company import Company

    return Company(symbol=symbol, source=provider)


def _fetch_from_provider(
    symbol: str,
    provider: str,
    start: date,
    end: date,
    *,
    page_size: int,
    max_pages: int,
) -> tuple[pd.DataFrame, str | None, bool]:
    """Fetch paginated news for a single ticker and provider.

    Returns the concatenated DataFrame, an optional error message, and whether
    provider pagination likely truncated the result set.
    """
    try:
        company = _make_company(symbol, provider)
    except Exception as exc:
        return pd.DataFrame(), str(exc), False

    provider_upper = provider.upper()

    if provider_upper == "VCI":
        try:
            df = company.news()
        except Exception as exc:
            return pd.DataFrame(), str(exc), False
        if df is None or df.empty:
            return pd.DataFrame(), None, False
        return _clean_provider_frame(df), None, False

    if provider_upper != "KBS":
        return pd.DataFrame(), f"Unsupported vnstock provider: {provider}", False

    pages: list[pd.DataFrame] = []
    for page in range(max_pages):
        try:
            df = company.news(page=page + 1, page_size=page_size)
        except Exception as exc:
            partial = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
            return partial, str(exc), False

        if df is None or df.empty:
            break

        df = _clean_provider_frame(df)
        pages.append(df)

        if len(df) < page_size:
            break

    if not pages:
        return pd.DataFrame(), None, False

    capped = len(pages) == max_pages and len(pages[-1]) == page_size

    return pd.concat(pages, ignore_index=True), None, capped


def _fetch_ticker_news(
    symbol: str,
    start: date,
    end: date,
    *,
    providers: list[str],
    page_size: int,
    max_pages: int,
) -> tuple[pd.DataFrame, str | None, int, list[str], bool]:
    """Fetch news for a single ticker via vnstock with provider fallback."""
    failures: list[str] = []
    for index, provider in enumerate(providers):
        df, error, capped = _fetch_from_provider(
            symbol,
            provider,
            start,
            end,
            page_size=page_size,
            max_pages=max_pages,
        )
        if error:
            failures.append(f"{provider}:{error}")
            logger.warning("vnstock %s failed for %s: %s", provider, symbol, error)
        elif not df.empty:
            return df, provider, index, failures, capped
        else:
            logger.info("vnstock %s returned no news for %s", provider, symbol)

    return pd.DataFrame(), None, 0, failures, False


def _first_present(row: pd.Series, candidates: tuple[str, ...]) -> str:
    """Return the first non-empty scalar for the given candidate columns."""
    for candidate in candidates:
        if candidate in row.index:
            value = _normalize_text(row.get(candidate))
            if value:
                return value
    return ""


def _normalize_article_dates(row: pd.Series, provider: str) -> tuple[str, str]:
    """Normalize provider-specific date fields to (date, published_at)."""
    provider_upper = provider.upper()
    if provider_upper == "VCI":
        timestamp_value = _first_present(row, ("published_at",))
        if timestamp_value:
            return _normalize_date_fields(timestamp_value)
        date_value = _first_present(
            row,
            ("public_date", "publicDate", "publish_date", "publishDate", "date"),
        )
        if date_value:
            return _normalize_date_fields(date_value)
        raise ValueError("missing VCI date field")

    if provider_upper == "KBS":
        timestamp_value = _first_present(row, ("publish_time",))
        if timestamp_value:
            return _normalize_date_fields(timestamp_value)
        date_value = _first_present(row, ("date", "publish_date", "publishDate"))
        if date_value:
            return _normalize_date_fields(date_value)
        raise ValueError("missing KBS date field")

    timestamp_value = _first_present(row, ("published_at", "publish_time"))
    if timestamp_value:
        return _normalize_date_fields(timestamp_value)
    date_value = _first_present(
        row,
        ("public_date", "publicDate", "publish_date", "publishDate", "date"),
    )
    if date_value:
        return _normalize_date_fields(date_value)
    raise ValueError("missing vnstock date field")


def _fallback_row_id(row: pd.Series, provider: str, row_number: int) -> str:
    """Build a stable row identifier when the provider omits the article URL."""
    provider_upper = provider.upper()
    if provider_upper == "VCI":
        candidates = ("news_id", "newsId", "id")
    elif provider_upper == "KBS":
        candidates = ("id", "news_id", "newsId")
    else:
        candidates = ("id", "news_id", "newsId")

    for candidate in candidates:
        if candidate in row.index:
            value = _normalize_text(row.get(candidate))
            if value:
                return value
    return str(row_number)


def _provider_url(
    row: pd.Series, provider: str, symbol: str, row_number: int
) -> tuple[str, bool]:
    """Return (url, is_real_url) for a normalized row."""
    url = _first_present(
        row, ("url", "news_source_link", "newsSourceLink", "source_url", "link")
    )
    if url:
        return url, True

    row_id = _fallback_row_id(row, provider, row_number)
    provider_slug = provider.lower()
    return f"vnstock://{provider_slug}/{symbol}/{row_id}", False


def _provider_title(row: pd.Series, provider: str) -> str:
    """Return the provider-appropriate title/headline field."""
    provider_upper = provider.upper()
    if provider_upper == "VCI":
        return _first_present(row, ("news_title", "newsTitle", "title", "headline"))
    if provider_upper == "KBS":
        return _first_present(row, ("head", "title", "news_title", "headline"))
    return _first_present(row, ("title", "news_title", "newsTitle", "head", "headline"))


def _provider_body(row: pd.Series, provider: str) -> str:
    """Return the provider-appropriate body/content field."""
    provider_upper = provider.upper()
    if provider_upper == "VCI":
        return _first_present(
            row,
            (
                "content",
                "news_full_content",
                "newsFullContent",
                "news_short_content",
                "newsShortContent",
                "description",
            ),
        )
    if provider_upper == "KBS":
        return _first_present(row, ("content", "description", "title"))
    return _first_present(
        row,
        (
            "content",
            "description",
            "body",
            "news_full_content",
            "newsFullContent",
        ),
    )


def _dataframe_to_articles(
    df: pd.DataFrame,
    symbol: str,
    provider: str,
) -> tuple[list[Article], _ArticleMetrics]:
    """Convert a vnstock news DataFrame to a list of ``Article`` objects."""
    articles: list[Article] = []
    metrics = _ArticleMetrics()
    if df.empty:
        return articles, metrics

    for row_number, (_, row) in enumerate(df.iterrows()):
        title = _provider_title(row, provider)
        if not title:
            continue

        try:
            iso_date, published_at = _normalize_article_dates(row, provider)
        except Exception:
            continue

        body = _provider_body(row, provider)
        if not body:
            body = title
            metrics.title_body_fallbacks += 1

        url, is_real_url = _provider_url(row, provider, symbol, row_number)
        if is_real_url:
            metrics.rows_with_real_urls += 1
        if published_at:
            metrics.rows_with_published_at += 1

        articles.append(
            Article(
                url=url,
                source="vnstock",
                category=f"Ticker:{symbol}",
                title=title,
                date=iso_date,
                published_at=published_at,
                body=body,
            )
        )

    if not articles:
        logger.warning(
            "vnstock %s DataFrame for %s yielded no normalizable rows: %s",
            provider,
            symbol,
            list(df.columns),
        )

    return articles, metrics


def run_vnstock(
    start: date,
    end: date,
    *,
    symbols: list[str] | None = None,
    providers: list[str] | None = None,
    page_size: int = VNSTOCK_PAGE_SIZE,
    max_pages: int = VNSTOCK_MAX_PAGES,
    discover_only: bool = False,
) -> SourceStats:
    import time

    """Fetch company news for configured symbols, filtered to ``[start, end]``.

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).
        symbols: Override the default symbol list from config.
        discover_only: If True, count articles without writing.

    Returns:
        Counters for the run.
    """
    stats = SourceStats()
    target_symbols = symbols or VNSTOCK_SYMBOLS
    target_providers = [
        provider.upper() for provider in (providers or VNSTOCK_PROVIDER_ORDER)
    ]
    output_file = Path(RAW_DATA_DIR) / SOURCE_OUTPUTS["vnstock"]
    all_articles: list[Article] = []

    for symbol in target_symbols:
        logger.info(
            "vnstock: fetching news for %s via %s", symbol, ",".join(target_providers)
        )
        df, provider_used, fallback_index, failures, capped = _fetch_ticker_news(
            symbol,
            start,
            end,
            providers=target_providers,
            page_size=page_size,
            max_pages=max_pages,
        )
        stats.provider_failures += len(failures)
        stats.fallback_uses += fallback_index

        # Add sleep between tickers to avoid rate limits
        time.sleep(2.0)

        if df.empty:
            logger.info(
                "vnstock: no news returned for %s after providers %s",
                symbol,
                ",".join(target_providers),
            )
            continue

        logger.info(
            "vnstock: %s returned %d raw rows for %s", provider_used, len(df), symbol
        )
        if capped:
            stats.capped_symbols += 1
            logger.warning(
                "vnstock: %s hit pagination cap for %s at %d rows (%d pages x %d)",
                provider_used,
                symbol,
                len(df),
                max_pages,
                page_size,
            )

        articles, metrics = _dataframe_to_articles(
            df, symbol, provider_used or "UNKNOWN"
        )
        stats.discovered_urls += len(articles)
        stats.rows_with_real_urls += metrics.rows_with_real_urls
        stats.rows_with_published_at += metrics.rows_with_published_at
        stats.title_body_fallbacks += metrics.title_body_fallbacks

        # Post-filter by date range
        for article in articles:
            try:
                article_date = parse_iso_date(article.date)
            except ValueError:
                stats.failed_pages += 1
                continue
            if article_date < start or article_date > end:
                stats.date_filter_skips += 1
                continue
            all_articles.append(article)
            stats.parsed_articles += 1

    if not discover_only and all_articles:
        append_articles(output_file, all_articles)

    logger.info(
        "vnstock: %d articles from %d symbols (%d in date range, %d provider failures, %d fallbacks, %d capped symbols, %d real URLs, %d timestamped rows, %d title-body fallbacks)",
        stats.discovered_urls,
        len(target_symbols),
        stats.parsed_articles,
        stats.provider_failures,
        stats.fallback_uses,
        stats.capped_symbols,
        stats.rows_with_real_urls,
        stats.rows_with_published_at,
        stats.title_body_fallbacks,
    )
    return stats
