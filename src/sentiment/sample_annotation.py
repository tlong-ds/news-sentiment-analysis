"""Export a review sample from the normalized training corpus."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.sentiment.common import TRAINING_REQUIRED_COLUMNS, validate_required_columns
from src.utils.io import read_parquet_table

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample normalized training rows for manual annotation."
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--sample-size", type=int, default=6000)
    parser.add_argument("--source-balance", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-file")
    return parser.parse_args()


def _category_frequency_bucket(series: pd.Series) -> pd.Series:
    counts = series.fillna("").astype(str).value_counts()
    return (
        series.fillna("")
        .astype(str)
        .map(
            lambda value: (
                "high"
                if counts.get(value, 0) >= 1000
                else "mid"
                if counts.get(value, 0) >= 100
                else "low"
            )
        )
    )


def _sample_stratified(
    df: pd.DataFrame, *, sample_size: int, seed: int
) -> pd.DataFrame:
    if df.empty or sample_size <= 0:
        return df.head(0)
    working = df.copy()
    published = pd.to_datetime(working["published_at"], errors="coerce")
    working["sample_year"] = published.dt.year.fillna(-1).astype(int)
    working["category_bucket"] = _category_frequency_bucket(working["category"])
    working["source_dataset"] = working.get("source_dataset", working["source"]).fillna(
        working["source"]
    )
    group_keys = ["source_dataset", "sample_year", "category_bucket"]
    groups = [group.copy() for _, group in working.groupby(group_keys, dropna=False)]
    groups.sort(key=lambda group: tuple(group.iloc[0][key] for key in group_keys))

    selected_parts: list[pd.DataFrame] = []
    seen_indices: set[int] = set()
    while len(seen_indices) < min(sample_size, len(working)):
        progressed = False
        for group in groups:
            remaining = group.loc[~group.index.isin(seen_indices)]
            if remaining.empty:
                continue
            chosen = remaining.sample(n=1, random_state=seed + len(seen_indices))
            selected_parts.append(chosen)
            seen_indices.add(int(chosen.index[0]))
            progressed = True
            if len(seen_indices) >= min(sample_size, len(working)):
                break
        if not progressed:
            break
    return (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else working.head(0)
    )


def sample_annotation_frame(
    df: pd.DataFrame, *, sample_size: int, seed: int, source_balance: bool = False
) -> pd.DataFrame:
    validate_required_columns(
        df, TRAINING_REQUIRED_COLUMNS, dataset_name="normalized training corpus"
    )
    available = df.copy()
    if "label" in available.columns:
        available = available[
            available["label"].isna()
            | (available["label"].astype(str).str.strip() == "")
        ]
    sample_n = min(sample_size, len(available))

    if source_balance and "source_dataset" in available.columns and sample_n:
        per_source = sample_n // max(1, available["source_dataset"].nunique())
        remainder = sample_n % max(1, available["source_dataset"].nunique())
        sample_parts: list[pd.DataFrame] = []
        for offset, (source_name, group) in enumerate(
            sorted(available.groupby("source_dataset"), key=lambda item: item[0])
        ):
            target = per_source + (1 if offset < remainder else 0)
            sample_parts.append(
                _sample_stratified(group, sample_size=target, seed=seed + offset)
            )
        sample = pd.concat(sample_parts, ignore_index=True)
        if len(sample) < sample_n:
            remaining = available.loc[
                ~available["article_id"].isin(sample["article_id"])
            ]
            extra = _sample_stratified(
                remaining, sample_size=sample_n - len(sample), seed=seed + 999
            )
            sample = pd.concat([sample, extra], ignore_index=True)
    else:
        sample = (
            _sample_stratified(available, sample_size=sample_n, seed=seed)
            if sample_n
            else available.head(0)
        )
    export_cols = [
        "article_id",
        "source",
        "category",
        "published_at",
        "title",
        "body_text",
        "input_text",
    ]
    if "source_dataset" in sample.columns:
        export_cols.append("source_dataset")
    sample = sample[export_cols].copy()
    sample["label"] = ""
    return sample.sort_values("article_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    df = read_parquet_table(args.input_file)
    sample = sample_annotation_frame(
        df,
        sample_size=args.sample_size,
        seed=args.seed,
        source_balance=args.source_balance,
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        sample.to_csv(output_path, index=False)
    else:
        sample.to_parquet(output_path, index=False)
    report_file = args.report_file or str(output_path.with_suffix(".report.json"))
    Path(report_file).write_text(
        json.dumps(
            {"sample_rows": int(len(sample)), "output_file": str(output_path)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %d annotation rows -> %s", len(sample), output_path)


if __name__ == "__main__":
    main()
