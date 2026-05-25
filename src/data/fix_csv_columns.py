"""Fix the column layout of CafeF news CSV after mixed-schema appending.

Older CafeF scrapes used a 6-column schema (missing ``published_at``).
The new scraper writes a 7-column schema. When resuming, new rows are
appended with 7 columns, but the header and older rows remain 6-column,
which corrupts the CSV file for standard parsers like Pandas.

This script scans the CSV row-by-row, pads the older 6-column rows to 7 columns
by inserting an empty string for ``published_at``, writes a 7-column header,
and outputs a clean, unified CSV.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

# Increase field size limit for large article bodies
csv.field_size_limit(sys.maxsize)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fix_csv(file_path: Path) -> None:
    temp_path = file_path.with_suffix(".csv.tmp")
    logger.info("Reading %s and writing to %s...", file_path, temp_path)

    header_7 = ["url", "source", "category", "title", "date", "published_at", "body"]

    total_rows = 0
    padded_rows = 0
    correct_rows = 0

    with (
        open(file_path, "r", encoding="utf-8") as f_in,
        open(temp_path, "w", encoding="utf-8", newline="") as f_out,
    ):
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Skip original header
        try:
            orig_header = next(reader)
            logger.info("Original Header: %s", orig_header)
        except StopIteration:
            logger.error("CSV file is empty!")
            return

        # Write new 7-column header
        writer.writerow(header_7)

        for row in reader:
            total_rows += 1
            if len(row) == 6:
                # Pad to 7 columns by inserting empty published_at at index 5
                row.insert(5, "")
                padded_rows += 1
            elif len(row) == 7:
                correct_rows += 1
            else:
                logger.warning(
                    "Row %d has unexpected column count (%d): %s",
                    total_rows,
                    len(row),
                    row[:2],
                )

            writer.writerow(row)
            if total_rows % 50000 == 0:
                logger.info("Processed %d rows...", total_rows)

    logger.info(
        "Finished fixing. Total rows: %d, Padded (old schema): %d, Correct (new schema): %d",
        total_rows,
        padded_rows,
        correct_rows,
    )

    # Rename temp to original
    temp_path.replace(file_path)
    logger.info("Successfully replaced original file.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix mixed-schema CSV files.")
    parser.add_argument(
        "--file",
        default="data/raw/news_VN_cafef.csv",
        help="Path to the CSV file to fix.",
    )
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        logger.error("File %s does not exist!", csv_path)
        sys.exit(1)

    fix_csv(csv_path)
