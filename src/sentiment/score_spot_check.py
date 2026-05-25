"""Score a manually labeled 50-row spot-check file."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from src.config import ANNOTATION_DATA_DIR

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score author labels against LLM consensus labels.")
    parser.add_argument("--input-file", default=f"{ANNOTATION_DATA_DIR}/spot_check_sample.parquet")
    parser.add_argument("--output-file", default=f"{ANNOTATION_DATA_DIR}/spot_check_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df = pd.read_parquet(args.input_file)
    required = {"final_label", "author_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Spot-check file missing columns: {missing}")
    scored = df[df["author_label"].fillna("").astype(str).ne("")].copy()
    if scored.empty:
        raise ValueError("No author labels found in spot-check file.")
    accuracy = float(scored["author_label"].astype(str).str.lower().eq(scored["final_label"].astype(str).str.lower()).mean())
    report = {
        "scored_rows": int(len(scored)),
        "accuracy": accuracy,
        "statement": f"Author spot-check of {len(scored)} randomly selected articles yielded {accuracy:.2%} agreement with LLM consensus labels.",
    }
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote spot-check results -> %s", args.output_file)


if __name__ == "__main__":
    main()
