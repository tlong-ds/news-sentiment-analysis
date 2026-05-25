"""Pipeline orchestrator — runs all supported sources and produces unified output.

Usage:
    python -m src.ingestion.pipeline --start 2015-01-01 --end 2024-12-31

Sources are executed in order:
    1. CafeF  — sitemap-based discovery + HTML parsing (primary)
    2. vnstock — ticker-level company news via API

Results are written per-source and merged into a deduplicated combined CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from src.config import RAW_DATA_DIR, SOURCE_OUTPUTS, START_DATE, END_DATE
from src.ingestion.base import Article, CSV_COLUMNS, Ledger, SourceStats
from src.utils.date_utils import parse_iso_date

# Increase CSV field size limit to handle very large article bodies
csv.field_size_limit(sys.maxsize)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # Fallback for systems where sys.maxsize exceeds C long limit
    csv.field_size_limit(2147483647)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese financial news ingestion pipeline."
    )
    parser.add_argument(
        "--start", default=START_DATE, help="Start date, YYYY-MM-DD."
    )
    parser.add_argument(
        "--end", default=END_DATE, help="End date, YYYY-MM-DD."
    )
    parser.add_argument(
        "--sources",
        default="cafef,vnstock",
        help="Comma-separated sources: cafef, vnstock.",
    )

    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Cap discovery pages per source for smoke runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the persistent URL ledger.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover URLs and report counts.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _merge_and_deduplicate(data_dir: Path) -> int:
    """Merge all per-source CSVs into a single deduplicated file.

    Deduplication is based on ``sha256(title_lower | date)[:16]``.

    Returns:
        Number of unique articles in the combined file.
    """
    combined_path = data_dir / "news_VN_2015_2024.csv"
    seen_hashes: set[str] = set()
    total = 0

    with combined_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for source_file in SOURCE_OUTPUTS.values():
            source_path = data_dir / source_file
            if not source_path.exists():
                continue
            with source_path.open("r", encoding="utf-8") as inp:
                reader = csv.DictReader(inp)
                for row in reader:
                    key = f"{row.get('title', '').strip().lower()}|{row.get('date', '')}"
                    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    writer.writerow(row)
                    total += 1

    logger.info("Merged %d unique articles → %s", total, combined_path)
    return total


def run_pipeline(args: argparse.Namespace) -> dict[str, SourceStats]:
    """Execute the full ingestion pipeline."""
    start = parse_iso_date(args.start)
    end = parse_iso_date(args.end)
    if start > end:
        raise ValueError("--start must be on or before --end")

    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    valid_sources = set(SOURCE_OUTPUTS.keys())
    unknown = sorted(set(sources) - valid_sources)
    if unknown:
        raise ValueError(f"Unknown sources: {', '.join(unknown)}")

    data_dir = Path(RAW_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Clean outputs unless resuming
    if not args.resume:
        for source in sources:
            (data_dir / SOURCE_OUTPUTS[source]).unlink(missing_ok=True)
        (data_dir / "news_VN_2015_2024.csv").unlink(missing_ok=True)

    ledger_path = data_dir / "news_scrape_ledger.json"
    ledger = Ledger.load(ledger_path) if args.resume else Ledger()
    results: dict[str, SourceStats] = {}

    for source in sources:
        logger.info("=" * 60)
        logger.info("Starting source: %s", source)
        logger.info("=" * 60)

        if source == "cafef":
            from src.ingestion.cafef_source import run_cafef

            stats = run_cafef(
                start,
                end,
                ledger=ledger,
                limit_pages=args.limit_pages,
                discover_only=args.discover_only,
            )
        elif source == "vnstock":
            from src.ingestion.vnstock_source import run_vnstock

            stats = run_vnstock(
                start,
                end,
                discover_only=args.discover_only,
            )
        else:

            logger.warning("Skipping unknown source: %s", source)
            continue

        results[source] = stats
        ledger.save(ledger_path)
        logger.info("%s stats: %s", source, asdict(stats))

    # Merge only when combining multiple sources into the unified corpus.
    if not args.discover_only and len(sources) > 1:
        _merge_and_deduplicate(data_dir)
    elif not args.discover_only:
        logger.info("Skipping combined merge for single-source run: %s", sources[0])

    ledger.save(ledger_path)
    return results


def main() -> None:
    setup_logging()
    args = parse_args()
    stats = run_pipeline(args)
    print(
        json.dumps(
            {source: asdict(item) for source, item in stats.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
