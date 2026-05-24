"""CafeF news source — sitemap-based discovery + article parsing.

CafeF publishes 5-day-block sitemaps with predictable URLs:
    ``cafef.vn/sitemaps/sitemaps-{YYYY}-{M}-{D1}-{D2}.xml``

This is the most reliable Vietnamese financial news source for historical
data collection.  Sitemaps are permissive (``Allow: /`` for all agents)
and return XML even for years-old content.

Article detail pages are static HTML, parseable with BeautifulSoup.
"""

from __future__ import annotations

import calendar
import logging
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    BUSINESS_KEYWORDS,
    CAFEF_BASE_URL,
    RAW_DATA_DIR,
    REQUEST_DELAY_SECONDS,
    SOURCE_OUTPUTS,
    USER_AGENT,
)
from src.ingestion.base import Article, Ledger, SourceStats, append_articles
from src.utils.date_utils import extract_date, parse_iso_date

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi,en;q=0.8",
        }
    )
    return session


def _fetch_text(session: requests.Session, url: str, timeout: int = 20) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _text_from_selectors(soup: BeautifulSoup, selectors: list[str]) -> list[str]:
    parts: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            text = _clean_text(node.get_text(" ", strip=True))
            if text:
                parts.append(text)
    return parts


def _infer_category(url: str, fallback: str = "Kinh doanh") -> str:
    for keyword, category in BUSINESS_KEYWORDS.items():
        if keyword in url:
            return category
    return fallback


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------

def _day_ranges(year: int, month: int) -> list[tuple[int, int]]:
    """Return the 5-day block ranges CafeF uses for a given month.

    The last segment adapts to the actual month length:
    Feb (non-leap) → 26-28, Feb (leap) → 26-29,
    30-day months → 26-30, 31-day months → 26-31.
    """
    _, last_day = calendar.monthrange(year, month)
    return [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, last_day)]


def cafef_sitemap_urls(
    start: date, end: date, *, limit_pages: int | None = None
) -> list[str]:
    """Discover CafeF business-article URLs via 5-day-block sitemaps.

    Args:
        start: Earliest date to collect.
        end: Latest date to collect.
        limit_pages: Cap on sitemap files fetched (for smoke tests).

    Returns:
        Sorted list of unique article URLs matching ``BUSINESS_KEYWORDS``.
    """
    session = _build_session()
    urls: set[str] = set()
    fetched = 0
    current = start.replace(day=1)
    while current <= end:
        for start_day, end_day in _day_ranges(current.year, current.month):
            sitemap_url = (
                f"https://cafef.vn/sitemaps/sitemaps-{current.year}-{current.month}-"
                f"{start_day}-{end_day}.xml"
            )
            if limit_pages is not None and fetched >= limit_pages:
                return sorted(urls)
            fetched += 1
            try:
                xml_text = _fetch_text(session, sitemap_url, timeout=15)
                root = ET.fromstring(xml_text)
            except Exception as exc:
                logger.debug("CafeF sitemap failed %s: %s", sitemap_url, exc)
                continue
            for node in root.iter():
                if node.tag.endswith("loc") and node.text:
                    url = node.text.strip()
                    if ".chn" in url and any(kw in url for kw in BUSINESS_KEYWORDS):
                        urls.add(url)
        next_month = current.replace(day=28) + timedelta(days=4)
        current = next_month.replace(day=1)
    return sorted(urls)


# ---------------------------------------------------------------------------
# Article parsing
# ---------------------------------------------------------------------------

def parse_cafef_article(url: str, html: str) -> Article | None:
    """Parse a CafeF article detail page into an ``Article``.

    Returns ``None`` when the page lacks a title, date, or body text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_node = soup.select_one("h1.title, h1.title_post, h1")
    title = _clean_text(title_node.get_text(" ", strip=True) if title_node else "")

    # Date
    date_node = soup.select_one(
        "meta[property='article:published_time'], meta[name='pubdate']"
    )
    date_text = date_node.get("content") if date_node else ""
    if not date_text:
        fallback_node = soup.select_one(".pdate, .td-post-date, .date")
        date_text = _clean_text(fallback_node.get_text(" ", strip=True) if fallback_node else "")

    # Body
    lead = _text_from_selectors(soup, [".sapo"])
    body = _text_from_selectors(
        soup,
        [
            ".detail-content p",
            ".knc-content p",
            "[data-role='content'] p",
            ".content_detail p",
            "#content_detail p",
        ],
    )

    parsed_date = extract_date(date_text)
    full_body = _clean_text(" ".join([*lead, *body]))
    if not title or not parsed_date or not full_body:
        return None

    return Article(
        url=url,
        source="cafef",
        category=_infer_category(url),
        title=title,
        date=parsed_date.isoformat(),
        published_at=pd.Timestamp(date_text).isoformat() if date_text else "",
        body=full_body,
    )


# ---------------------------------------------------------------------------
# Full pipeline: discover → parse → write
# ---------------------------------------------------------------------------

def run_cafef(
    start: date,
    end: date,
    *,
    ledger: Ledger | None = None,
    limit_pages: int | None = None,
    discover_only: bool = False,
) -> SourceStats:
    """Run the full CafeF ingestion pipeline.

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).
        ledger: Optional resume ledger for deduplication.
        limit_pages: Cap sitemap fetches for smoke testing.
        discover_only: If True, only discover URLs without parsing.

    Returns:
        Counters for the run.
    """
    from pathlib import Path

    stats = SourceStats()
    ledger = ledger or Ledger()
    session = _build_session()

    logger.info("CafeF: discovering article URLs (%s to %s)", start, end)
    urls = cafef_sitemap_urls(start, end, limit_pages=limit_pages)
    deduped = sorted(set(urls))
    stats.discovered_urls = len(deduped)

    if not deduped:
        logger.info("CafeF: no URLs discovered")
        return stats

    if discover_only:
        logger.info("CafeF: discover-only found %d URLs", len(deduped))
        return stats

    urls_to_process = []
    for url in deduped:
        if url in ledger.completed_urls:
            stats.skipped_duplicates += 1
        else:
            urls_to_process.append(url)

    logger.info("CafeF: parsing %d articles using thread pool", len(urls_to_process))
    output_file = Path(RAW_DATA_DIR) / SOURCE_OUTPUTS["cafef"]
    batch: list[Article] = []

    def fetch_and_parse(target_url: str) -> tuple[str, Article | None, Exception | None]:
        # Local session per thread for safety
        thread_session = _build_session()
        try:
            html = _fetch_text(thread_session, target_url)
            return target_url, parse_cafef_article(target_url, html), None
        except Exception as exc:
            return target_url, None, exc

    processed = 0
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(fetch_and_parse, url): url for url in urls_to_process}
        
        for future in as_completed(futures):
            processed += 1
            if processed % 1000 == 0:
                logger.info("CafeF: processed %d / %d articles", processed, len(urls_to_process))
                
            url, article, exc = future.result()
            
            if exc:
                logger.debug("CafeF detail failed %s: %s", url, exc)
                ledger.failed_urls.add(url)
                stats.failed_pages += 1
                continue

            if article is None:
                ledger.failed_urls.add(url)
                stats.failed_pages += 1
                continue

            try:
                article_date = parse_iso_date(article.date)
                if article_date < start or article_date > end:
                    stats.date_filter_skips += 1
                    ledger.completed_urls.add(url)
                    continue
            except ValueError:
                stats.failed_pages += 1
                ledger.failed_urls.add(url)
                continue

            batch.append(article)
            ledger.completed_urls.add(url)
            stats.parsed_articles += 1

            # Periodically write to CSV
            if len(batch) >= 100:
                append_articles(output_file, batch)
                batch.clear()

    if batch:
        append_articles(output_file, batch)
        
    logger.info("CafeF: %d articles parsed, %d failed", stats.parsed_articles, stats.failed_pages)
    return stats
