"""Tests for the modular ingestion pipeline.

Tests article parsing for CafeF (the primary source), shared date
extraction, and category inference.  VnExpress parsing is retained for
the existing test fixture but is no longer part of the production pipeline.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from src.ingestion.base import Article
from src.ingestion.cafef_source import (
    _infer_category,
    cafef_sitemap_urls,
    parse_cafef_article,
)
from src.ingestion.vnstock_source import (
    _dataframe_to_articles,
    _fetch_ticker_news,
    run_vnstock,
)
from src.utils.date_utils import extract_date

FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CafeF article parsing
# ---------------------------------------------------------------------------

def test_parse_cafef_article_fixture():
    article = parse_cafef_article(
        "https://cafef.vn/thi-truong-chung-khoan/example-20150103.chn",
        read_fixture("cafef_article.html"),
    )

    assert article is not None
    assert article.source == "cafef"
    assert article.category == "Chứng khoán"
    assert article.title == "Cổ phiếu ngân hàng khởi sắc"
    assert article.date == "2015-01-03"
    assert article.published_at == "2015-01-03T09:00:00+07:00"
    assert "chính sách tiền tệ" in article.body


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

def test_extract_date_formats():
    assert extract_date("2015-01-02T08:30:00+07:00") == date(2015, 1, 2)
    assert extract_date("Thứ sáu, 02/01/2015, 08:30") == date(2015, 1, 2)
    assert extract_date("02-01-2015") == date(2015, 1, 2)


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------

def test_infer_category_prefers_business_keywords():
    from src.config import BUSINESS_KEYWORDS

    assert BUSINESS_KEYWORDS["bat-dong-san"] == "Bất động sản"
    assert _infer_category("https://example.test/bat-dong-san/foo") == "Bất động sản"


# ---------------------------------------------------------------------------
# CafeF sitemap discovery (mocked)
# ---------------------------------------------------------------------------

def test_cafef_sitemap_url_extraction(monkeypatch):
    sitemap = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://cafef.vn/thi-truong-chung-khoan/a-20150101.chn</loc></url>
      <url><loc>https://cafef.vn/doi-song/a-20150101.chn</loc></url>
      <url><loc>https://cafef.vn/doanh-nghiep/b-20150101.chn</loc></url>
    </urlset>
    """

    monkeypatch.setattr(
        "src.ingestion.cafef_source._fetch_text",
        lambda *_args, **_kwargs: sitemap,
    )

    urls = cafef_sitemap_urls(date(2015, 1, 1), date(2015, 1, 5), limit_pages=1)

    assert urls == [
        "https://cafef.vn/doanh-nghiep/b-20150101.chn",
        "https://cafef.vn/thi-truong-chung-khoan/a-20150101.chn",
    ]


# ---------------------------------------------------------------------------
# Article title hash (deduplication)
# ---------------------------------------------------------------------------

def test_article_title_hash_deterministic():
    a1 = Article(
        url="https://example.com/1",
        source="cafef",
        category="Chứng khoán",
        title="Test Title",
        date="2024-01-01",
        published_at="2024-01-01T09:00:00+07:00",
        body="Body text",
    )
    a2 = Article(
        url="https://example.com/2",  # Different URL
        source="vnstock",  # Different source
        category="Kinh doanh",
        title="Test Title",
        date="2024-01-01",
        published_at="",
        body="Different body",
    )
    # Same title + date → same hash (for dedup)
    assert a1.title_hash() == a2.title_hash()

    a3 = Article(
        url="https://example.com/3",
        source="cafef",
        category="Chứng khoán",
        title="Different Title",
        date="2024-01-01",
        published_at="2024-01-01T11:00:00+07:00",
        body="Body text",
    )
    assert a1.title_hash() != a3.title_hash()


def test_vnstock_dataframe_to_articles_normalizes_missing_values():
    df = pd.DataFrame(
        [
            {
                "news_title": "Sample headline",
                "public_date": "2024-12-31T08:30:00",
                "content": None,
                "url": None,
                "id": 42,
            }
        ]
    )

    articles, metrics = _dataframe_to_articles(df, "VHM", "VCI")

    assert len(articles) == 1
    assert articles[0].url == "vnstock://vci/VHM/42"
    assert articles[0].published_at == "2024-12-31T08:30:00"
    assert articles[0].body == "Sample headline"
    assert metrics.rows_with_published_at == 1
    assert metrics.rows_with_real_urls == 0
    assert metrics.title_body_fallbacks == 1


def test_vnstock_dataframe_to_articles_leaves_published_at_blank_for_date_only_rows():
    df = pd.DataFrame(
        [
            {
                "news_title": "Sample headline",
                "public_date": "2024-12-31",
                "news_full_content": "Body",
                "news_source_link": "https://example.test/article",
            }
        ]
    )

    articles, metrics = _dataframe_to_articles(df, "VHM", "VCI")

    assert len(articles) == 1
    assert articles[0].date == "2024-12-31"
    assert articles[0].published_at == ""
    assert metrics.rows_with_real_urls == 1
    assert metrics.rows_with_published_at == 0


def test_vnstock_dataframe_to_articles_supports_vci_camelcase_backcompat():
    df = pd.DataFrame(
        [
            {
                "newsTitle": "Sample headline",
                "publicDate": "2024-12-31T08:30:00+07:00",
                "newsFullContent": "Body",
                "newsSourceLink": "https://example.test/article",
            }
        ]
    )

    articles, metrics = _dataframe_to_articles(df, "VHM", "VCI")

    assert len(articles) == 1
    assert articles[0].published_at == "2024-12-31T08:30:00+07:00"
    assert articles[0].body == "Body"
    assert metrics.rows_with_real_urls == 1


def test_vnstock_dataframe_to_articles_supports_kbs_documented_fields():
    df = pd.DataFrame(
        [
            {
                "head": "KBS headline",
                "publish_time": "2024-12-31T08:30:00.123456+07:00",
                "url": "https://example.test/kbs-article",
                "title": "Optional body/title field",
            }
        ]
    )

    articles, metrics = _dataframe_to_articles(df, "VHM", "KBS")

    assert len(articles) == 1
    assert articles[0].title == "KBS headline"
    assert articles[0].date == "2024-12-31"
    assert articles[0].published_at == "2024-12-31T08:30:00.123456+07:00"
    assert articles[0].body == "Optional body/title field"
    assert metrics.rows_with_real_urls == 1


def test_vnstock_dataframe_to_articles_generates_unique_fallback_urls():
    df = pd.DataFrame(
        [
            {
                "head": "Headline 1",
                "publish_time": "2024-12-31T08:30:00",
                "id": "abc",
            },
            {
                "head": "Headline 2",
                "publish_time": "2024-12-31T09:30:00",
                "id": "xyz",
            },
        ]
    )

    articles, _ = _dataframe_to_articles(df, "VHM", "KBS")

    assert [article.url for article in articles] == [
        "vnstock://kbs/VHM/abc",
        "vnstock://kbs/VHM/xyz",
    ]


def test_vnstock_fetch_ticker_news_uses_fallback_provider(monkeypatch):
    calls: list[str] = []

    def fake_fetch_from_provider(symbol, provider, start, end, *, page_size, max_pages):
        calls.append(provider)
        if provider == "VCI":
            return pd.DataFrame(), "provider down", False
        return pd.DataFrame(
            [
                {
                    "head": f"{symbol} headline",
                    "publish_time": "2024-12-31T08:30:00+07:00",
                    "url": "https://example.test/article",
                }
            ]
        ), None, False

    monkeypatch.setattr("src.ingestion.vnstock_source._fetch_from_provider", fake_fetch_from_provider)

    df, provider_used, fallback_index, failures, capped = _fetch_ticker_news(
        "VHM",
        date(2024, 1, 1),
        date(2024, 12, 31),
        providers=["VCI", "KBS"],
        page_size=50,
        max_pages=10,
    )

    assert calls == ["VCI", "KBS"]
    assert provider_used == "KBS"
    assert fallback_index == 1
    assert len(failures) == 1
    assert len(df) == 1
    assert capped is False


def test_vnstock_fetch_from_provider_marks_pagination_cap(monkeypatch):
    class FakeCompany:
        def __init__(self):
            self.calls = 0

        def news(self, *, page, page_size):
            self.calls += 1
            return pd.DataFrame(
                [
                    {"head": f"headline-{self.calls}-{idx}", "publish_time": "2024-12-31T08:30:00"}
                    for idx in range(page_size)
                ]
            )

    monkeypatch.setattr("src.ingestion.vnstock_source._make_company", lambda symbol, provider: FakeCompany())

    df, error, capped = __import__("src.ingestion.vnstock_source", fromlist=["_fetch_from_provider"])._fetch_from_provider(
        "VHM",
        "KBS",
        date(2024, 1, 1),
        date(2024, 12, 31),
        page_size=2,
        max_pages=3,
    )

    assert error is None
    assert capped is True
    assert len(df) == 6


def test_run_vnstock_counts_provider_failures_and_fallbacks(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingestion.vnstock_source.RAW_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("src.ingestion.vnstock_source.SOURCE_OUTPUTS", {"vnstock": "news_VN_vnstock.csv"})
    monkeypatch.setattr("src.ingestion.vnstock_source.append_articles", lambda *args, **kwargs: None)

    def fake_sleep(_seconds):
        return None

    def fake_fetch_ticker_news(symbol, start, end, *, providers, page_size, max_pages):
        if symbol == "VHM":
            return pd.DataFrame([{"title": "ignored"}]), "KBS", 1, ["VCI:down"], True
        return pd.DataFrame(), None, 0, ["VCI:down", "KBS:empty"], False

    monkeypatch.setitem(__import__("sys").modules, "time", type("FakeTime", (), {"sleep": staticmethod(fake_sleep)}))
    monkeypatch.setattr("src.ingestion.vnstock_source._fetch_ticker_news", fake_fetch_ticker_news)
    monkeypatch.setattr(
        "src.ingestion.vnstock_source._dataframe_to_articles",
        lambda df, symbol, provider: (
            [
                Article(
                    url=f"vnstock://{provider.lower()}/{symbol}/1",
                    source="vnstock",
                    category=f"Ticker:{symbol}",
                    title=f"{symbol} headline",
                    date="2024-12-31",
                    published_at="2024-12-31T08:30:00",
                    body="Body",
                )
            ],
            type(
                "Metrics",
                (),
                {
                    "rows_with_real_urls": 0,
                    "rows_with_published_at": 1,
                    "title_body_fallbacks": 0,
                },
            )(),
        ) if not df.empty else ([], type("Metrics", (), {"rows_with_real_urls": 0, "rows_with_published_at": 0, "title_body_fallbacks": 0})()),
    )

    stats = run_vnstock(
        date(2024, 1, 1),
        date(2024, 12, 31),
        symbols=["VHM", "VCB"],
        providers=["VCI", "KBS"],
        page_size=50,
        max_pages=10,
    )

    assert stats.discovered_urls == 1
    assert stats.parsed_articles == 1
    assert stats.provider_failures == 3
    assert stats.fallback_uses == 1
    assert stats.capped_symbols == 1
    assert stats.rows_with_published_at == 1
