"""Shared types and protocols for the news ingestion pipeline.

Every source module produces ``Article`` objects that the pipeline orchestrator
collects, deduplicates, and writes to CSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

CSV_COLUMNS = ["url", "source", "category", "title", "date", "published_at", "body"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Article:
    """A single news article with metadata."""

    url: str
    source: str
    category: str
    title: str
    date: str  # ISO format YYYY-MM-DD
    body: str
    published_at: str = ""  # ISO timestamp when the source exposes intraday time

    def title_hash(self) -> str:
        """Deterministic hash for deduplication by title + date."""
        key = f"{self.title.strip().lower()}|{self.date}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class NewsSource(Protocol):
    """Interface that every ingestion source must satisfy."""

    name: str

    def discover(self, start: date, end: date, *, limit_pages: int | None = None) -> list[str]:
        """Return a list of article identifiers (URLs, IDs, etc.)."""
        ...

    def parse(self, identifier: str) -> Article | None:
        """Fetch and parse a single article.  Return ``None`` on failure."""
        ...


# ---------------------------------------------------------------------------
# Resume ledger (shared across sources)
# ---------------------------------------------------------------------------

@dataclass
class Ledger:
    """Tracks completed and failed URLs for resumable runs."""

    completed_urls: set[str] = field(default_factory=set)
    failed_urls: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> Ledger:
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
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------

@dataclass
class SourceStats:
    """Counters for a single source run."""

    discovered_urls: int = 0
    parsed_articles: int = 0
    skipped_duplicates: int = 0
    date_filter_skips: int = 0
    failed_pages: int = 0
    provider_failures: int = 0
    fallback_uses: int = 0
    capped_symbols: int = 0
    rows_with_real_urls: int = 0
    rows_with_published_at: int = 0
    title_body_fallbacks: int = 0


def append_articles(path: Path, articles: list[Article]) -> None:
    """Append a batch of ``Article`` objects to a CSV file."""
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
