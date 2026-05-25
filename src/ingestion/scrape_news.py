"""Hybrid news scraper for Vietnamese business news.

Discovery uses source-specific strategies:
- CafeF historical sitemaps are fetched over HTTP (5-day block sitemaps).
- VnExpress uses Playwright-driven search (timkiem.vnexpress.net) with a
  static listing fallback.  Yearly article sitemaps exist but are blocked
  server-side for all automated clients.
- Vietstock sitemaps are fetched over HTTP when accessible, falling back to
  Playwright-driven search pages.
- Article detail pages are fetched over HTTP and parsed with BeautifulSoup.
"""

from __future__ import annotations

import argparse
import calendar
import asyncio
import csv
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import sys

# Add project root to sys.path to allow running this script directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import RAW_DATA_DIR

try:  # Playwright is required for full discovery, but parser tests should not need it.
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover - exercised only in lean environments.
    PlaywrightTimeoutError = TimeoutError
    async_playwright = None


DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-12-31"
CSV_COLUMNS = ["url", "source", "category", "title", "date", "body"]
REQUEST_DELAY_SECONDS = 0.6
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

SOURCE_OUTPUTS = {
    "vnexpress": "news_VN_vnexpress.csv",
    "cafef": "news_VN_cafef.csv",
    "vietstock": "news_VN_vietstock.csv",
}

BUSINESS_KEYWORDS = {
    "chung-khoan": "Chứng khoán",
    "thi-truong-chung-khoan": "Chứng khoán",
    "doanh-nghiep": "Doanh nghiệp",
    "tai-chinh": "Tài chính",
    "tai-chinh-ngan-hang": "Tài chính ngân hàng",
    "kinh-te": "Kinh tế",
    "kinh-te-vi-mo": "Vĩ mô",
    "vi-mo": "Vĩ mô",
    "bat-dong-san": "Bất động sản",
    "kinh-doanh": "Kinh doanh",
}

VNEXPRESS_LISTINGS = [
    ("https://vnexpress.net/kinh-doanh", "Kinh doanh"),
    ("https://vnexpress.net/bat-dong-san", "Bất động sản"),
]
VNEXPRESS_STATIC_MAX_PAGES = 5

VIETSTOCK_SEARCH_TERMS = ["chứng khoán", "doanh nghiệp", "tài chính", "bất động sản"]


@dataclass(slots=True)
class Article:
    url: str
    source: str
    category: str
    title: str
    date: str
    body: str


@dataclass
class SourceStats:
    discovered_urls: int = 0
    parsed_articles: int = 0
    skipped_duplicates: int = 0
    date_filter_skips: int = 0
    failed_pages: int = 0
    skipped_blank_discovery: int = 0


@dataclass
class Ledger:
    completed_urls: set[str] = field(default_factory=set)
    failed_urls: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            completed_urls=set(raw.get("completed_urls", [])),
            failed_urls=set(raw.get("failed_urls", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed_urls": sorted(self.completed_urls),
            "failed_urls": sorted(self.failed_urls),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Vietnamese business news.")
    parser.add_argument(
        "--start", default=DEFAULT_START, help="Start date, YYYY-MM-DD."
    )
    parser.add_argument("--end", default=DEFAULT_END, help="End date, YYYY-MM-DD.")
    parser.add_argument(
        "--sources",
        default="vnexpress,cafef",
        help="Comma-separated sources: vnexpress,cafef,vietstock.",
    )
    parser.add_argument(
        "--headful", action="store_true", help="Run Playwright with a visible browser."
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Cap discovery pages/sitemaps per source for smoke runs.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the persistent URL ledger."
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover URLs and report counts.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_blocks(start: date, end: date, days: int) -> Iterable[tuple[date, date]]:
    current = start
    while current <= end:
        block_end = min(current + timedelta(days=days - 1), end)
        yield current, block_end
        current = block_end + timedelta(days=1)


def unix_seconds(day: date, end_of_day: bool = False) -> int:
    clock = datetime_time(23, 59, 59) if end_of_day else datetime_time(0, 0, 0)
    return int(datetime.combine(day, clock).timestamp())


def date_from_unix_seconds(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def output_path(filename: str) -> Path:
    return Path(RAW_DATA_DIR) / filename


def prepare_outputs(sources: list[str], resume: bool) -> None:
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    if resume:
        return
    for source in sources:
        output_path(SOURCE_OUTPUTS[source]).unlink(missing_ok=True)
    output_path("news_VN_2015_2024.csv").unlink(missing_ok=True)
    output_path("news_scrape_ledger.json").unlink(missing_ok=True)


def append_articles(path: Path, articles: list[Article]) -> None:
    if not articles:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for article in articles:
            writer.writerow(asdict(article))


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def text_from_selectors(soup: BeautifulSoup, selectors: list[str]) -> list[str]:
    parts: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                parts.append(text)
    return parts


def extract_date(value: str | None) -> date | None:
    if not value:
        return None
    patterns = [
        (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
        (r"(\d{1,2}/\d{1,2}/\d{4})", "%d/%m/%Y"),
        (r"(\d{1,2}-\d{1,2}-\d{4})", "%d-%m-%Y"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, value)
        if match:
            try:
                return datetime.strptime(match.group(1), fmt).date()
            except ValueError:
                continue
    return None


def infer_category(url: str, fallback: str = "Kinh doanh") -> str:
    for keyword, category in BUSINESS_KEYWORDS.items():
        if keyword in url:
            return category
    return fallback


def normalize_url(base_url: str, href: str | None) -> str | None:
    if not href:
        return None
    href = href.strip()
    if not href or href.startswith(("javascript:", "mailto:")):
        return None
    return urljoin(base_url, href).split("#", 1)[0]


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi,en;q=0.8",
        }
    )
    return session


def fetch_text(session: requests.Session, url: str, timeout: int = 20) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def parse_article_html(
    source: str, url: str, html: str, category: str = ""
) -> Article | None:
    soup = BeautifulSoup(html, "html.parser")
    if source == "vnexpress":
        title = clean_text(
            soup.select_one("h1.title-detail, h1.title_post, h1").get_text(
                " ", strip=True
            )
            if soup.select_one("h1.title-detail, h1.title_post, h1")
            else ""
        )
        date_node = soup.select_one(
            "meta[name='pubdate'], meta[property='article:published_time']"
        )
        date_text = date_node.get("content") if date_node else ""
        date_text = date_text or clean_text(
            soup.select_one("span.date, .date").get_text(" ", strip=True)
            if soup.select_one("span.date, .date")
            else ""
        )
        lead = text_from_selectors(soup, ["p.description"])
        body = text_from_selectors(
            soup, ["article.fck_detail p", ".fck_detail > p", ".fck_detail p"]
        )
    elif source == "cafef":
        title = clean_text(
            soup.select_one("h1.title, h1.title_post, h1").get_text(" ", strip=True)
            if soup.select_one("h1.title, h1.title_post, h1")
            else ""
        )
        date_node = soup.select_one(
            "meta[property='article:published_time'], meta[name='pubdate']"
        )
        date_text = date_node.get("content") if date_node else ""
        date_text = date_text or clean_text(
            soup.select_one(".pdate, .td-post-date, .date").get_text(" ", strip=True)
            if soup.select_one(".pdate, .td-post-date, .date")
            else ""
        )
        lead = text_from_selectors(soup, [".sapo"])
        body = text_from_selectors(
            soup,
            [
                ".detail-content p",
                ".knc-content p",
                "[data-role='content'] p",
                ".content_detail p",
                "#content_detail p",
            ],
        )
    elif source == "vietstock":
        title = clean_text(
            soup.select_one("h1.article-title, h1.title-detail, h1").get_text(
                " ", strip=True
            )
            if soup.select_one("h1.article-title, h1.title-detail, h1")
            else ""
        )
        date_node = soup.select_one(
            "meta[itemprop='datePublished'], meta[property='article:published_time']"
        )
        date_text = date_node.get("content") if date_node else ""
        date_text = date_text or clean_text(
            soup.select_one(".date, .article-date, .date-published").get_text(
                " ", strip=True
            )
            if soup.select_one(".date, .article-date, .date-published")
            else ""
        )
        lead = []
        body = text_from_selectors(
            soup,
            [
                ".article-content p",
                ".article-body p",
                ".article-content div",
                ".fck_detail p",
            ],
        )
    else:
        raise ValueError(f"Unsupported source: {source}")

    parsed_date = extract_date(date_text)
    full_body = clean_text(" ".join([*lead, *body]))
    if not title or not parsed_date or not full_body:
        return None
    return Article(
        url=url,
        source=source,
        category=category or infer_category(url),
        title=title,
        date=parsed_date.isoformat(),
        body=full_body,
    )


def _cafef_day_ranges(year: int, month: int) -> list[tuple[int, int]]:
    """Return the 5-day block ranges CafeF uses for a given month.

    The last segment adapts to the actual month length:
    Feb (non-leap) → 26-28, Feb (leap) → 26-29,
    30-day months → 26-30, 31-day months → 26-31.
    """
    _, last_day = calendar.monthrange(year, month)
    return [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, last_day)]


def cafef_sitemap_urls(
    start: date, end: date, limit_pages: int | None = None
) -> list[str]:
    session = build_session()
    urls: set[str] = set()
    fetched = 0
    current = start.replace(day=1)
    while current <= end:
        for start_day, end_day in _cafef_day_ranges(current.year, current.month):
            sitemap_url = (
                f"https://cafef.vn/sitemaps/sitemaps-{current.year}-{current.month}-"
                f"{start_day}-{end_day}.xml"
            )
            if limit_pages is not None and fetched >= limit_pages:
                return sorted(urls)
            fetched += 1
            try:
                xml_text = fetch_text(session, sitemap_url, timeout=15)
                root = ET.fromstring(xml_text)
            except Exception as exc:
                logging.debug("CafeF sitemap failed %s: %s", sitemap_url, exc)
                continue
            for node in root.iter():
                if node.tag.endswith("loc") and node.text:
                    url = node.text.strip()
                    if ".chn" in url and any(
                        keyword in url for keyword in BUSINESS_KEYWORDS
                    ):
                        urls.add(url)
        next_month = current.replace(day=28) + timedelta(days=4)
        current = next_month.replace(day=1)
    return sorted(urls)


async def discover_vnexpress_urls(
    start: date, end: date, headful: bool, limit_pages: int | None
) -> list[str]:
    """Primary: Playwright search.  Fallback: static category listings.

    VnExpress yearly article sitemaps exist but are blocked server-side for
    all automated clients (302 → homepage regardless of cookies/UA), so
    Playwright-driven search is the reliable primary strategy.
    """
    if async_playwright is None:
        logging.warning(
            "Playwright is not installed; falling back to static VnExpress listings."
        )
        return discover_vnexpress_static(start, end, limit_pages)

    playwright_urls: set[str] = set()
    pages_seen = 0
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not headful)
        context = await browser.new_context(user_agent=USER_AGENT, locale="vi-VN")
        page = await context.new_page()
        try:
            for block_start, block_end in date_blocks(start, end, 31):
                if limit_pages is not None and pages_seen >= limit_pages:
                    break
                search_url = (
                    "https://timkiem.vnexpress.net/"
                    "?q=kinh%20doanh&media_type=all"
                    f"&fromdate={unix_seconds(block_start)}"
                    f"&todate={unix_seconds(block_end, end_of_day=True)}"
                    "&cate_code=&search_f=title,tag_list&date_format=custom"
                )
                try:
                    await page.goto(
                        search_url, wait_until="domcontentloaded", timeout=45_000
                    )
                    await page.wait_for_timeout(1200)
                    pages_seen += 1
                    while True:
                        hrefs = await page.locator(
                            "article[data-url], h3.title-news a, h2.title-news a, article a"
                        ).evaluate_all(
                            """(nodes) => nodes.map((node) => {
                                if (node.dataset && node.dataset.url) return node.dataset.url;
                                return node.href || node.getAttribute('href');
                            }).filter(Boolean)"""
                        )
                        for href in hrefs:
                            if "vnexpress.net" in href and re.search(
                                r"\d+\.html$", href
                            ):
                                playwright_urls.add(href)
                        if limit_pages is not None and pages_seen >= limit_pages:
                            break
                        next_link = page.locator(
                            "a.next-page, a.btn-page.next-page, a[rel='next']"
                        ).first
                        if await next_link.count() == 0:
                            break
                        await next_link.click(timeout=5000)
                        await page.wait_for_load_state(
                            "domcontentloaded", timeout=15_000
                        )
                        await page.wait_for_timeout(900)
                        pages_seen += 1
                    if limit_pages is not None and pages_seen >= limit_pages:
                        break
                except PlaywrightTimeoutError as exc:
                    logging.info(
                        "VnExpress search timed out for %s to %s: %s",
                        block_start,
                        block_end,
                        exc,
                    )
        finally:
            await browser.close()

    if not playwright_urls:
        logging.warning(
            "VnExpress Playwright discovery returned no URLs; using static listing fallback."
        )
        return discover_vnexpress_static(start, end, limit_pages)
    return sorted(playwright_urls)


def discover_vnexpress_static(
    start: date, end: date, limit_pages: int | None
) -> list[str]:
    session = build_session()
    urls: set[str] = set()
    pages_seen = 0
    for listing_url, _category in VNEXPRESS_LISTINGS:
        seen_in_range = False
        max_pages = limit_pages or VNEXPRESS_STATIC_MAX_PAGES
        for page_number in range(1, max_pages + 1):
            if limit_pages is not None and pages_seen >= limit_pages:
                return sorted(urls)
            url = listing_url if page_number == 1 else f"{listing_url}-p{page_number}"
            pages_seen += 1
            try:
                html = fetch_text(session, url, timeout=15)
            except Exception as exc:
                logging.debug("VnExpress listing failed %s: %s", url, exc)
                continue
            soup = BeautifulSoup(html, "html.parser")
            page_dates: list[date] = []
            for article_node in soup.select("article[data-publishtime]"):
                article_date = date_from_unix_seconds(
                    article_node.get("data-publishtime")
                )
                if article_date:
                    page_dates.append(article_date)
                if article_date and article_date > end:
                    continue
                if article_date and article_date < start:
                    continue
                anchor = article_node.select_one(
                    "h3.title-news a, h2.title-news a, a[href]"
                )
                normalized = normalize_url(
                    "https://vnexpress.net", anchor.get("href") if anchor else None
                )
                if (
                    normalized
                    and "vnexpress.net" in normalized
                    and re.search(r"\d+\.html$", normalized)
                ):
                    urls.add(normalized)
                    seen_in_range = True
            # Some listing variants omit data-publishtime; keep a conservative fallback
            # only after we have reached pages that overlap the requested date range.
            if seen_in_range and not page_dates:
                for anchor in soup.select(
                    "h3.title-news a, h2.title-news a, article a"
                ):
                    normalized = normalize_url(
                        "https://vnexpress.net", anchor.get("href")
                    )
                    if (
                        normalized
                        and "vnexpress.net" in normalized
                        and re.search(r"\d+\.html$", normalized)
                    ):
                        urls.add(normalized)
            if page_dates and max(page_dates) < start:
                break
            if not page_dates and page_number > 5 and not seen_in_range:
                break
            if page_number >= VNEXPRESS_STATIC_MAX_PAGES and not seen_in_range:
                logging.warning(
                    "VnExpress static listing only exposed recent pages for %s; no URLs in %s to %s.",
                    listing_url,
                    start,
                    end,
                )
                break
    return sorted(urls)


def vnexpress_listing_date_range(
    page_number: int,
    listing_url: str = "https://vnexpress.net/kinh-doanh",
) -> tuple[date | None, date | None, int]:
    session = build_session()
    url = listing_url if page_number == 1 else f"{listing_url}-p{page_number}"
    html = fetch_text(session, url, timeout=15)
    soup = BeautifulSoup(html, "html.parser")
    dates = [
        item
        for item in (
            date_from_unix_seconds(node.get("data-publishtime"))
            for node in soup.select("article[data-publishtime]")
        )
        if item is not None
    ]
    if not dates:
        return None, None, 0
    return min(dates), max(dates), len(dates)


def vietstock_sitemap_urls(limit_pages: int | None = None) -> list[str]:
    """Discover Vietstock article URLs via sitemap.xml over HTTP.

    Vietstock's sitemap may return 403 to some clients.  The function uses
    browser-like headers from ``build_session()`` and falls back gracefully.
    """
    session = build_session()
    urls: set[str] = set()
    try:
        index_xml = fetch_text(session, "https://vietstock.vn/sitemap.xml", timeout=20)
        index_root = ET.fromstring(index_xml)
    except Exception as exc:
        logging.info("Vietstock sitemap index not accessible: %s", exc)
        return []

    child_urls = [
        node.text.strip()
        for node in index_root.iter()
        if node.tag.endswith("loc") and node.text
    ]
    fetched = 0
    for child_url in child_urls:
        if limit_pages is not None and fetched >= limit_pages:
            break
        fetched += 1
        try:
            child_xml = fetch_text(session, child_url, timeout=20)
            child_root = ET.fromstring(child_xml)
        except Exception as exc:
            logging.debug("Vietstock child sitemap failed %s: %s", child_url, exc)
            continue
        for node in child_root.iter():
            if node.tag.endswith("loc") and node.text:
                url = node.text.strip()
                if re.search(r"\d+\.htm$", url) and any(
                    keyword in url for keyword in BUSINESS_KEYWORDS
                ):
                    urls.add(url)
    if urls:
        logging.info("Vietstock sitemap discovery found %d business URLs", len(urls))
    return sorted(urls)


async def discover_vietstock_urls(
    start: date, end: date, headful: bool, limit_pages: int | None
) -> list[str]:
    """Primary: sitemap over HTTP.  Fallback: Playwright search."""
    urls = vietstock_sitemap_urls(limit_pages)
    if urls:
        return urls

    logging.info("Vietstock sitemap unavailable; falling back to Playwright search.")
    if async_playwright is None:
        logging.warning("Playwright is not installed; skipping Vietstock discovery.")
        return []

    playwright_urls: set[str] = set()
    pages_seen = 0
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not headful)
        context = await browser.new_context(user_agent=USER_AGENT, locale="vi-VN")
        page = await context.new_page()
        try:
            for term in VIETSTOCK_SEARCH_TERMS:
                for block_start, block_end in date_blocks(start, end, 90):
                    if limit_pages is not None and pages_seen >= limit_pages:
                        break
                    search_url = (
                        "https://vietstock.vn/tim-kiem.htm"
                        f"?q={term.replace(' ', '+')}"
                        f"&fromDate={block_start.strftime('%d/%m/%Y')}"
                        f"&toDate={block_end.strftime('%d/%m/%Y')}"
                    )
                    try:
                        await page.goto(
                            search_url, wait_until="networkidle", timeout=45_000
                        )
                        await page.wait_for_timeout(1500)
                        pages_seen += 1
                        hrefs = await page.locator(
                            "div.news-item h2 a, div.news-item h4 a, article a"
                        ).evaluate_all(
                            "(nodes) => nodes.map((node) => node.href).filter(Boolean)"
                        )
                    except PlaywrightTimeoutError as exc:
                        logging.info(
                            "Vietstock search timed out for %s to %s: %s",
                            block_start,
                            block_end,
                            exc,
                        )
                        continue
                    for href in hrefs:
                        if "vietstock.vn" in href and re.search(r"\d+\.htm$", href):
                            playwright_urls.add(href)
                if limit_pages is not None and pages_seen >= limit_pages:
                    break
        finally:
            await browser.close()

    if not playwright_urls:
        logging.warning(
            "Vietstock discovery produced no URLs; source will be skipped as best-effort."
        )
    return sorted(playwright_urls)


async def discover_urls(
    source: str, start: date, end: date, headful: bool, limit_pages: int | None
) -> list[str]:
    if source == "cafef":
        return cafef_sitemap_urls(start, end, limit_pages)
    if source == "vnexpress":
        return await discover_vnexpress_urls(start, end, headful, limit_pages)
    if source == "vietstock":
        return await discover_vietstock_urls(start, end, headful, limit_pages)
    raise ValueError(f"Unknown source: {source}")


def parse_articles(
    source: str,
    urls: Iterable[str],
    start: date,
    end: date,
    ledger: Ledger,
    stats: SourceStats,
) -> None:
    session = build_session()
    source_batch: list[Article] = []
    combined_batch: list[Article] = []
    source_file = output_path(SOURCE_OUTPUTS[source])
    combined_file = output_path("news_VN_2015_2024.csv")

    for url in urls:
        if url in ledger.completed_urls:
            stats.skipped_duplicates += 1
            continue
        try:
            html = fetch_text(session, url)
            article = parse_article_html(source, url, html, infer_category(url))
        except Exception as exc:
            logging.debug("Detail fetch/parse failed %s: %s", url, exc)
            ledger.failed_urls.add(url)
            stats.failed_pages += 1
            continue

        if article is None:
            ledger.failed_urls.add(url)
            stats.failed_pages += 1
            continue
        article_date = parse_iso_date(article.date)
        if article_date < start or article_date > end:
            stats.date_filter_skips += 1
            ledger.completed_urls.add(url)
            continue

        source_batch.append(article)
        combined_batch.append(article)
        ledger.completed_urls.add(url)
        stats.parsed_articles += 1

        if len(source_batch) >= 25:
            append_articles(source_file, source_batch)
            append_articles(combined_file, combined_batch)
            source_batch.clear()
            combined_batch.clear()
        time.sleep(REQUEST_DELAY_SECONDS)

    append_articles(source_file, source_batch)
    append_articles(combined_file, combined_batch)


async def run_scrape(args: argparse.Namespace) -> dict[str, SourceStats]:
    start = parse_iso_date(args.start)
    end = parse_iso_date(args.end)
    if start > end:
        raise ValueError("--start must be on or before --end")

    sources = [
        source.strip().lower() for source in args.sources.split(",") if source.strip()
    ]
    unknown = sorted(set(sources) - set(SOURCE_OUTPUTS))
    if unknown:
        raise ValueError(f"Unknown sources: {', '.join(unknown)}")

    prepare_outputs(sources, args.resume)
    ledger_path = output_path("news_scrape_ledger.json")
    ledger = Ledger.load(ledger_path) if args.resume else Ledger()
    results: dict[str, SourceStats] = {}

    for source in sources:
        stats = SourceStats()
        results[source] = stats
        logging.info("Discovering %s URLs", source)
        urls = await discover_urls(source, start, end, args.headful, args.limit_pages)
        deduped_urls = sorted(set(urls))
        stats.discovered_urls = len(deduped_urls)
        if not deduped_urls:
            stats.skipped_blank_discovery += 1
            ledger.save(ledger_path)
            logging.info("%s produced no URLs; continuing", source)
            continue
        if args.discover_only:
            logging.info("%s discover-only found %s URLs", source, len(deduped_urls))
            ledger.save(ledger_path)
            continue
        logging.info("Parsing %s %s articles", len(deduped_urls), source)
        parse_articles(source, deduped_urls, start, end, ledger, stats)
        ledger.save(ledger_path)
        logging.info("%s stats: %s", source, asdict(stats))

    ledger.save(ledger_path)
    return results


def main() -> None:
    setup_logging()
    args = parse_args()
    stats = asyncio.run(run_scrape(args))
    print(
        json.dumps(
            {source: asdict(item) for source, item in stats.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
