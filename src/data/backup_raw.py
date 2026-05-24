"""Backup the raw CafeF news CSV before any pipeline overwrites.

Usage::

    python -m src.data.backup_raw [--src data/raw/news_VN_cafef.csv]

The backup is written to the same directory with a ``_backup_YYYYMMDD`` suffix
so it is clearly identifiable as a point-in-time snapshot.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import date
from pathlib import Path

from src.config import RAW_DATA_DIR, SOURCE_OUTPUTS

logger = logging.getLogger(__name__)


def backup_raw_csv(
    src_path: Path | str | None = None,
    *,
    today: date | None = None,
) -> Path:
    """Copy the raw CafeF CSV to a timestamped backup path.

    Args:
        src_path: Source file to back up.  Defaults to ``data/raw/news_VN_cafef.csv``.
        today: Override date for the backup suffix (useful in tests).

    Returns:
        Absolute path of the backup file.

    Raises:
        FileNotFoundError: If ``src_path`` does not exist.
    """
    if src_path is None:
        src_path = Path(RAW_DATA_DIR) / SOURCE_OUTPUTS["cafef"]
    src = Path(src_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    stamp = (today or date.today()).strftime("%Y%m%d")
    backup_name = f"{src.stem}_backup_{stamp}{src.suffix}"
    dest = src.parent / backup_name

    if dest.exists():
        logger.info("Backup already exists, skipping copy: %s", dest)
        return dest

    logger.info("Copying %s → %s", src, dest)
    shutil.copy2(src, dest)
    size_mb = dest.stat().st_size / 1_048_576
    logger.info("Backup written: %s (%.1f MB)", dest, size_mb)
    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Back up the raw CafeF CSV before pipeline overwrites."
    )
    parser.add_argument(
        "--src",
        default=None,
        help="Source CSV to back up (default: data/raw/news_VN_cafef.csv).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    dest = backup_raw_csv(args.src)
    print(f"Backup path: {dest}")


if __name__ == "__main__":
    main()
