"""ViFiC (Vietnamese Financial Corpus) dataset loader.

ViFiC is a Kaggle dataset containing ~160,490 financial news articles
(2010–2025) from VnExpress, CafeF, VnEconomy, and others.  The data is
in sentence-level plain text format.

This module loads and processes the dataset into ``Article`` objects.
Because ViFiC is sentence-level text without structured per-article
metadata, it is primarily useful as supplementary training data or for
topic-model validation — **not** as the primary sentiment pipeline input.

Download: ``kaggle datasets download -d <dataset-slug>``
Place extracted files under: ``data/vific/``
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

from src.config import RAW_DATA_DIR, SOURCE_OUTPUTS, VIFIC_DATA_DIR
from src.ingestion.base import Article, SourceStats, append_articles
from src.utils.date_utils import extract_date, parse_iso_date

logger = logging.getLogger(__name__)


def _find_vific_files() -> list[Path]:
    """Locate ViFiC text files in the data directory."""
    vific_dir = Path(VIFIC_DATA_DIR)
    if not vific_dir.exists():
        logger.warning(
            "ViFiC data directory not found: %s. "
            "Download from Kaggle and place files there.",
            vific_dir,
        )
        return []
    # ViFiC may be a single large text file or split by source/year (or subfolders)
    files = sorted(vific_dir.rglob("*.txt")) + sorted(vific_dir.rglob("*.csv"))
    if not files:
        logger.warning("No .txt or .csv files found in %s", vific_dir)
    return files


def _parse_vific_txt(path: Path, start: date, end: date) -> list[Article]:
    """Parse a ViFiC plain-text file into articles.

    ViFiC format is sentence-per-line.  We attempt to reconstruct article
    boundaries by looking for blank-line separators or date markers.
    """
    articles: list[Article] = []
    current_lines: list[str] = []
    current_date: date | None = None
    line_count = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            stripped = line.strip()

            # Blank line often separates articles in sentence-level corpora
            if not stripped:
                if current_lines and current_date:
                    body = " ".join(current_lines)
                    title = current_lines[0][:200]  # First sentence as title
                    if start <= current_date <= end:
                        articles.append(
                            Article(
                                url=f"vific://{path.name}#L{line_count}",
                                source="vific",
                                category="Tài chính",  # Default for financial corpus
                                title=title,
                                date=current_date.isoformat(),
                                body=body,
                            )
                        )
                current_lines = []
                current_date = None
                continue

            # Try to extract a date from each line
            if current_date is None:
                extracted = extract_date(stripped)
                if extracted:
                    current_date = extracted

            current_lines.append(stripped)

    # Flush remaining
    if current_lines and current_date and start <= current_date <= end:
        body = " ".join(current_lines)
        articles.append(
            Article(
                url=f"vific://{path.name}#L{line_count}",
                source="vific",
                category="Tài chính",
                title=current_lines[0][:200],
                date=current_date.isoformat(),
                body=body,
            )
        )

    return articles


def _parse_vific_csv(path: Path, start: date, end: date) -> list[Article]:
    """Parse a ViFiC CSV file into articles (if the dataset uses CSV format)."""
    import csv

    articles: list[Article] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try common column names
            title = row.get("title", row.get("Title", "")).strip()
            body = row.get("content", row.get("body", row.get("text", ""))).strip()
            raw_date = row.get("date", row.get("publish_date", row.get("Date", ""))).strip()
            source_name = row.get("source", "vific").strip()
            category = row.get("category", "Tài chính").strip()
            url = row.get("url", row.get("link", f"vific://{path.name}")).strip()

            if not title or not body:
                continue

            parsed_date = extract_date(raw_date)
            if not parsed_date or parsed_date < start or parsed_date > end:
                continue

            articles.append(
                Article(
                    url=url,
                    source="vific",
                    category=category,
                    title=title,
                    date=parsed_date.isoformat(),
                    body=body,
                )
            )

    return articles


def run_vific(
    start: date,
    end: date,
    *,
    discover_only: bool = False,
) -> SourceStats:
    """Load ViFiC dataset and filter to the requested date range.

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).
        discover_only: If True, count articles without writing.

    Returns:
        Counters for the run.
    """
    stats = SourceStats()
    files = _find_vific_files()
    if not files:
        logger.info("ViFiC: no data files found, skipping")
        return stats

    output_file = Path(RAW_DATA_DIR) / SOURCE_OUTPUTS["vific"]
    all_articles: list[Article] = []

    for file_path in files:
        logger.info("ViFiC: processing %s", file_path.name)
        if file_path.suffix == ".csv":
            articles = _parse_vific_csv(file_path, start, end)
        else:
            articles = _parse_vific_txt(file_path, start, end)

        stats.discovered_urls += len(articles)
        all_articles.extend(articles)

    stats.parsed_articles = len(all_articles)

    if not discover_only and all_articles:
        append_articles(output_file, all_articles)

    logger.info("ViFiC: %d articles loaded from %d files", stats.parsed_articles, len(files))
    return stats
