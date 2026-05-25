"""Sample a stratified ViFiC subset for dual-LLM annotation."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

import pandas as pd

from src.config import ANNOTATION_DATA_DIR, VIFIC_NORMALIZED_DIR
from src.sentiment.common import split_period_bucket

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a ViFiC annotation subset.")
    parser.add_argument("--input-file", default=f"{VIFIC_NORMALIZED_DIR}/vific_input.parquet")
    parser.add_argument("--output-file", default=f"{ANNOTATION_DATA_DIR}/vific_annotation_sample.parquet")
    parser.add_argument("--report-file", default=f"{ANNOTATION_DATA_DIR}/vific_annotation_sample_report.json")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def assign_length_bin(token_count: int) -> str:
    if token_count < 80:
        return "short"
    if token_count < 150:
        return "medium"
    return "long"


def build_strata(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["period_bucket"] = split_period_bucket(working["date"])
    working["length_bin"] = working["token_count"].map(assign_length_bin)
    category = working["category"].replace("", "uncategorized").fillna("uncategorized")
    working["sample_stratum"] = (
        category.astype(str)
        + "|"
        + working["period_bucket"].astype(str)
        + "|"
        + working["length_bin"].astype(str)
    )
    return working


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        sampled = build_strata(df).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return sampled

    working = build_strata(df)
    sampled_parts: list[pd.DataFrame] = []
    length_targets = {"short": 0.20, "medium": 0.50, "long": 0.30}
    total_target_counts = {
        key: int(round(sample_size * ratio))
        for key, ratio in length_targets.items()
    }
    drift = sample_size - sum(total_target_counts.values())
    total_target_counts["medium"] += drift

    for offset, (length_bin, target_count) in enumerate(total_target_counts.items()):
        length_df = working[working["length_bin"] == length_bin]
        if length_df.empty or target_count <= 0:
            continue
        counts = length_df["sample_stratum"].value_counts().sort_index()
        allocations = (counts / counts.sum() * target_count).round().astype(int).clip(lower=1)
        for inner_offset, (stratum, target) in enumerate(allocations.items()):
            group = length_df[length_df["sample_stratum"] == stratum]
            take = min(target, len(group))
            sampled_parts.append(group.sample(n=take, random_state=seed + offset + inner_offset))

    sampled = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["article_id"])
    if len(sampled) < sample_size:
        remainder = working[~working["article_id"].isin(sampled["article_id"])]
        needed = min(sample_size - len(sampled), len(remainder))
        if needed > 0:
            sampled = pd.concat(
                [sampled, remainder.sample(n=needed, random_state=seed + 999)],
                ignore_index=True,
            )

    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return sampled.reset_index(drop=True)


def build_sampling_report(full_df: pd.DataFrame, sampled_df: pd.DataFrame) -> dict:
    full = build_strata(full_df)
    sampled = sampled_df.copy()
    full_length = full["length_bin"].value_counts(normalize=True).to_dict()
    sampled_length = sampled["length_bin"].value_counts(normalize=True).to_dict()
    return {
        "full_rows": int(len(full)),
        "sampled_rows": int(len(sampled)),
        "length_bin_distribution_full": full_length,
        "length_bin_distribution_sampled": sampled_length,
        "category_distribution_full": full["category"].replace("", "uncategorized").value_counts(normalize=True).head(20).to_dict(),
        "category_distribution_sampled": sampled["category"].replace("", "uncategorized").value_counts(normalize=True).head(20).to_dict(),
        "period_distribution_full": full["period_bucket"].value_counts(normalize=True).to_dict(),
        "period_distribution_sampled": sampled["period_bucket"].value_counts(normalize=True).to_dict(),
        "target_length_mix": {"short": 0.20, "medium": 0.50, "long": 0.30},
        "sampled_length_mix_deviation": {
            key: float(sampled_length.get(key, 0.0) - target)
            for key, target in {"short": 0.20, "medium": 0.50, "long": 0.30}.items()
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df = pd.read_parquet(args.input_file)
    sampled = stratified_sample(df, sample_size=args.sample_size, seed=args.seed)
    sampled.to_parquet(args.output_file, index=False)
    report = build_sampling_report(df, sampled)
    with open(args.report_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    logger.info("Sampled %d rows -> %s", len(sampled), args.output_file)


if __name__ == "__main__":
    main()
