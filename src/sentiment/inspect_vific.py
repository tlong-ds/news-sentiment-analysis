"""Inspect ViFiC structure before normalization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import FINETUNES_DATA_DIR, VIFIC_NORMALIZED_DIR

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a ViFiC source file and emit a schema summary.")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-file", default=f"{VIFIC_NORMALIZED_DIR}/vific_schema_summary.json")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def _default_vific_input() -> str:
    candidates = sorted(Path(FINETUNES_DATA_DIR).rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No ViFiC CSV found under data/fine-tunes/. Pass --input-file explicitly.")
    return str(candidates[0])


def inspect_vific(df: pd.DataFrame, top_n: int = 20) -> dict:
    columns = [str(column) for column in df.columns]
    lower_cols = {column.lower() for column in df.columns}
    date_col = next((column for column in df.columns if str(column).lower() in {"date", "publish_date"}), None)
    category_col = next((column for column in df.columns if str(column).lower() == "category"), None)
    source_col = next((column for column in df.columns if str(column).lower() == "source"), None)
    label_cols = [column for column in df.columns if "label" in str(column).lower() or "sentiment" in str(column).lower()]
    split_cols = [column for column in df.columns if "split" in str(column).lower()]

    summary = {
        "row_count": int(len(df)),
        "columns": columns,
        "has_separate_title": "title" in lower_cols,
        "has_separate_body": bool({"body", "content", "text"} & lower_cols),
        "category_column": str(category_col) if category_col else None,
        "source_column": str(source_col) if source_col else None,
        "label_like_columns": [str(column) for column in label_cols],
        "split_like_columns": [str(column) for column in split_cols],
    }
    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        summary["date_range"] = {
            "min": str(parsed.min()) if parsed.notna().any() else None,
            "max": str(parsed.max()) if parsed.notna().any() else None,
        }
    if category_col is not None:
        summary["category_distribution_top"] = (
            df[category_col].fillna("").astype(str).value_counts().head(top_n).to_dict()
        )
    if source_col is not None:
        summary["source_distribution_top"] = (
            df[source_col].fillna("").astype(str).value_counts().head(top_n).to_dict()
        )
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    input_file = args.input_file or _default_vific_input()
    df = pd.read_csv(input_file)
    summary = inspect_vific(df, top_n=args.top_n)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote ViFiC inspection summary -> %s", output_path)


if __name__ == "__main__":
    main()
