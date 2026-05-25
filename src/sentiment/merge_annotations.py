"""Merge bootstrap and reviewed sentiment labels back into the normalized corpus."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.sentiment.common import (
    LABELED_REQUIRED_COLUMNS,
    TRAINING_REQUIRED_COLUMNS,
    assign_splits,
    normalize_label,
    validate_required_columns,
)
from src.utils.io import read_table

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge external sentiment labels into the normalized training corpus."
    )
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument("--annotations-file", required=True)
    parser.add_argument("--reviewed-annotations-file")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--report-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--allow-low-confidence", action="store_true")
    return parser.parse_args()


def _normalize_annotation_frame(
    annotations_df: pd.DataFrame,
    *,
    dataset_name: str,
    confidence_threshold: float,
    allow_low_confidence: bool,
    reviewed: bool,
) -> pd.DataFrame:
    validate_required_columns(
        annotations_df, {"article_id", "label"}, dataset_name=dataset_name
    )
    if annotations_df["article_id"].duplicated().any():
        raise ValueError(f"{dataset_name} contains duplicate article_id values.")
    working = annotations_df.copy()
    if (
        working["label"].isna().any()
        or (working["label"].astype(str).str.strip() == "").any()
    ):
        raise ValueError(f"{dataset_name} contains missing labels.")
    working["label"] = working["label"].astype(str).map(normalize_label)
    if "confidence" in working.columns:
        working["confidence"] = pd.to_numeric(working["confidence"], errors="coerce")
        if working["confidence"].isna().any():
            raise ValueError(f"{dataset_name} contains non-numeric confidence values.")
        if not allow_low_confidence and not reviewed:
            working = working.loc[working["confidence"] >= confidence_threshold].copy()
    else:
        working["confidence"] = 1.0 if reviewed else None
    working["label_source"] = "reviewed" if reviewed else "bootstrap"
    return working


def merge_annotation_frames(
    corpus_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    *,
    reviewed_df: pd.DataFrame | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    confidence_threshold: float = 0.8,
    allow_low_confidence: bool = False,
) -> pd.DataFrame:
    validate_required_columns(
        corpus_df, TRAINING_REQUIRED_COLUMNS, dataset_name="normalized training corpus"
    )
    if corpus_df["article_id"].duplicated().any():
        raise ValueError(
            "Normalized training corpus contains duplicate article_id values."
        )

    bootstrap = _normalize_annotation_frame(
        annotations_df,
        dataset_name="annotation file",
        confidence_threshold=confidence_threshold,
        allow_low_confidence=allow_low_confidence,
        reviewed=False,
    )
    review = None
    if reviewed_df is not None:
        review = _normalize_annotation_frame(
            reviewed_df,
            dataset_name="reviewed annotation file",
            confidence_threshold=confidence_threshold,
            allow_low_confidence=True,
            reviewed=True,
        )

    merged = corpus_df.merge(
        bootstrap[["article_id", "label", "confidence", "label_source"]],
        on="article_id",
        how="left",
    )
    if review is not None:
        merged = merged.merge(
            review[["article_id", "label", "confidence", "label_source"]].rename(
                columns={
                    "label": "reviewed_label",
                    "confidence": "reviewed_confidence",
                    "label_source": "reviewed_label_source",
                }
            ),
            on="article_id",
            how="left",
        )
        reviewed_mask = merged["reviewed_label"].notna()
        merged.loc[reviewed_mask, "label"] = merged.loc[reviewed_mask, "reviewed_label"]
        merged.loc[reviewed_mask, "confidence"] = merged.loc[
            reviewed_mask, "reviewed_confidence"
        ]
        merged.loc[reviewed_mask, "label_source"] = merged.loc[
            reviewed_mask, "reviewed_label_source"
        ]
        merged = merged.drop(
            columns=["reviewed_label", "reviewed_confidence", "reviewed_label_source"]
        )

    merged = merged.loc[merged["label"].notna()].copy()
    if merged.empty:
        raise ValueError(
            "No article_id overlap between normalized corpus and annotation inputs."
        )
    merged["split"] = assign_splits(
        merged,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    ordered = LABELED_REQUIRED_COLUMNS + [
        column
        for column in ["source_dataset", "confidence", "label_source"]
        if column in merged.columns
    ]
    merged = merged[ordered].copy()
    validate_required_columns(
        merged, LABELED_REQUIRED_COLUMNS, dataset_name="labeled corpus"
    )
    return merged.sort_values(["split", "article_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    corpus_df = read_table(args.corpus_file)
    annotations_df = read_table(args.annotations_file)
    reviewed_df = (
        read_table(args.reviewed_annotations_file)
        if args.reviewed_annotations_file
        else None
    )
    merged = merge_annotation_frames(
        corpus_df,
        annotations_df,
        reviewed_df=reviewed_df,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        confidence_threshold=args.confidence_threshold,
        allow_low_confidence=args.allow_low_confidence,
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    report_file = args.report_file or str(output_path.with_suffix(".report.json"))
    report = {
        "rows": int(len(merged)),
        "labels": merged["label"].value_counts().to_dict(),
        "splits": merged["split"].value_counts().to_dict(),
    }
    if "source_dataset" in merged.columns:
        report["split_source_datasets"] = pd.crosstab(
            merged["split"], merged["source_dataset"]
        ).to_dict(orient="index")
    if "label_source" in merged.columns:
        report["label_sources"] = merged["label_source"].value_counts().to_dict()
    Path(report_file).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Merged %d labeled rows -> %s", len(merged), output_path)


if __name__ == "__main__":
    main()
