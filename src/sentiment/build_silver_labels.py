"""Parse raw LLM responses and build the silver-label training dataset."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config import ANNOTATION_DATA_DIR
from src.sentiment.common import LABEL_TO_ID, SENTIMENT_LABELS, read_jsonl

logger = logging.getLogger(__name__)


def compute_confusion_matrix(left: pd.Series, right: pd.Series, labels: list[str]) -> np.ndarray:
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    index = {label: idx for idx, label in enumerate(labels)}
    for lhs, rhs in zip(left, right):
        matrix[index[lhs], index[rhs]] += 1
    return matrix


def compute_cohen_kappa(left: pd.Series, right: pd.Series, labels: list[str]) -> float:
    matrix = compute_confusion_matrix(left, right, labels)
    total = matrix.sum()
    if total == 0:
        return 0.0
    observed = np.trace(matrix) / total
    row_marginals = matrix.sum(axis=1) / total
    col_marginals = matrix.sum(axis=0) / total
    expected = float(np.dot(row_marginals, col_marginals))
    if expected >= 1.0:
        return 1.0
    return float((observed - expected) / (1 - expected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a silver-label sentiment dataset.")
    parser.add_argument("--sample-file", default=f"{ANNOTATION_DATA_DIR}/vific_annotation_sample.csv")
    parser.add_argument("--llm-a-file", default=f"{ANNOTATION_DATA_DIR}/llm_a_raw_responses.jsonl")
    parser.add_argument("--llm-b-file", default=f"{ANNOTATION_DATA_DIR}/llm_b_raw_responses.jsonl")
    parser.add_argument("--merged-output", default=f"{ANNOTATION_DATA_DIR}/vific_llm_labels.csv")
    parser.add_argument("--dataset-output", default=f"{ANNOTATION_DATA_DIR}/sentiment_labeled_full.csv")
    parser.add_argument("--metrics-output", default=f"{ANNOTATION_DATA_DIR}/silver_label_metrics.json")
    parser.add_argument("--spot-check-output", default=f"{ANNOTATION_DATA_DIR}/spot_check_sample.csv")
    parser.add_argument("--disagreements-output", default=f"{ANNOTATION_DATA_DIR}/vific_disagreement_cases.csv")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--allow-low-kappa", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty response text.")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Response does not contain JSON.")
    return json.loads(text[start : end + 1])


def _normalize_record(record: dict, prefix: str) -> dict:
    payload = _extract_json(record.get("response_text", ""))
    label = str(payload["label"]).strip().lower()
    if label not in LABEL_TO_ID:
        raise ValueError(f"Unsupported label: {label}")
    confidence = float(payload["confidence"])
    return {
        "article_id": record["article_id"],
        f"{prefix}_label": label,
        f"{prefix}_confidence": confidence,
        f"{prefix}_reason": str(payload.get("reason", "")).strip(),
        f"{prefix}_model": record.get("model", ""),
        "annotation_timestamp": record.get("timestamp", ""),
    }


def parse_response_records(records: Iterable[dict], prefix: str) -> pd.DataFrame:
    normalized: list[dict] = []
    for record in records:
        try:
            normalized.append(_normalize_record(record, prefix))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Skipping malformed %s record for %s: %s", prefix, record.get("article_id"), exc)
    if not normalized:
        raise ValueError(f"No valid {prefix} annotation records were parsed.")
    return pd.DataFrame(normalized).drop_duplicates(subset=["article_id"], keep="last")


def build_consensus_table(
    sample_df: pd.DataFrame,
    llm_a_df: pd.DataFrame,
    llm_b_df: pd.DataFrame,
    confidence_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    merged = sample_df.merge(llm_a_df, on="article_id", how="inner").merge(llm_b_df, on="article_id", how="inner")
    merged["agreement"] = merged["llm_a_label"] == merged["llm_b_label"]
    merged["confidence_pass"] = (
        merged["llm_a_confidence"].ge(confidence_threshold)
        & merged["llm_b_confidence"].ge(confidence_threshold)
    )
    merged["final_label"] = merged["llm_a_label"].where(merged["agreement"] & merged["confidence_pass"])

    matrix = compute_confusion_matrix(merged["llm_a_label"], merged["llm_b_label"], SENTIMENT_LABELS)
    kappa = compute_cohen_kappa(merged["llm_a_label"], merged["llm_b_label"], SENTIMENT_LABELS)
    per_class_agreement = {}
    for label in SENTIMENT_LABELS:
        mask = merged["llm_a_label"].eq(label) | merged["llm_b_label"].eq(label)
        if mask.any():
            per_class_agreement[label] = float(
                merged.loc[mask, "llm_a_label"].eq(merged.loc[mask, "llm_b_label"]).mean()
            )
        else:
            per_class_agreement[label] = 0.0
    agreement_only_rows = int(merged["agreement"].sum())
    retained_rows = int(merged["final_label"].notna().sum())
    metrics = {
        "kappa": float(kappa),
        "confusion_matrix": matrix.tolist(),
        "labels": SENTIMENT_LABELS,
        "agreement_rows": agreement_only_rows,
        "confidence_threshold": confidence_threshold,
        "retained_rows": retained_rows,
        "per_class_agreement": per_class_agreement,
    }
    return merged, metrics


def split_labeled_dataset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    retained = df[df["final_label"].notna()].copy()
    if retained.empty:
        raise ValueError("No rows passed agreement/confidence filtering.")

    retained["label"] = retained["final_label"]
    retained["label_encoded"] = retained["label"].map(LABEL_TO_ID)
    retained["source_corpus"] = "vific"
    retained["label_source"] = "llm_agreement"

    if retained["label"].value_counts().min() < 3 or len(retained) < 9:
        retained = retained.sort_values(["label", "article_id"]).reset_index(drop=True)
        split_cycle = ["train", "train", "train", "val", "test", "train"]
        retained["split"] = [split_cycle[idx % len(split_cycle)] for idx in range(len(retained))]
        output = retained
    else:
        grouped_parts: list[pd.DataFrame] = []
        for _, group in retained.groupby("label", sort=True):
            shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n_rows = len(shuffled)
            train_end = max(1, int(round(n_rows * 0.70)))
            val_end = max(train_end + 1, int(round(n_rows * 0.85)))
            shuffled.loc[: train_end - 1, "split"] = "train"
            shuffled.loc[train_end : val_end - 1, "split"] = "val"
            shuffled.loc[val_end:, "split"] = "test"
            grouped_parts.append(shuffled)
        output = pd.concat(grouped_parts, ignore_index=True)
    keep_cols = [
        "article_id",
        "source_corpus",
        "input_text",
        "input_text_segmented",
        "label",
        "label_encoded",
        "label_source",
        "llm_a_label",
        "llm_b_label",
        "llm_a_confidence",
        "llm_b_confidence",
        "split",
    ]
    return output[keep_cols].sort_values(["split", "article_id"]).reset_index(drop=True)


def build_stage4_metrics(metrics: dict, merged: pd.DataFrame, threshold_used: float, threshold_fallback_triggered: bool) -> dict:
    kappa = metrics["kappa"]
    if kappa >= 0.65:
        kappa_status = "proceed"
    elif kappa >= 0.50:
        kappa_status = "reannotate_disagreements"
    else:
        kappa_status = "revise_prompt"
    return {
        **metrics,
        "confidence_threshold": threshold_used,
        "confidence_threshold_fallback_triggered": threshold_fallback_triggered,
        "kappa_status": kappa_status,
        "agreement_only_rows": int(merged["agreement"].sum()),
        "agreement_plus_confidence_rows": int(merged["final_label"].notna().sum()),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sample_df = pd.read_csv(args.sample_file)
    llm_a_df = parse_response_records(read_jsonl(args.llm_a_file), "llm_a")
    llm_b_df = parse_response_records(read_jsonl(args.llm_b_file), "llm_b")
    merged, metrics = build_consensus_table(
        sample_df,
        llm_a_df,
        llm_b_df,
        confidence_threshold=args.confidence_threshold,
    )
    threshold_used = args.confidence_threshold
    threshold_fallback_triggered = False
    if metrics["retained_rows"] < 2500 and args.confidence_threshold > 0.70:
        threshold_used = 0.70
        threshold_fallback_triggered = True
        merged, metrics = build_consensus_table(
            sample_df,
            llm_a_df,
            llm_b_df,
            confidence_threshold=threshold_used,
        )
    dataset = split_labeled_dataset(merged, seed=args.seed)
    disagreements = merged[~merged["agreement"]].copy()
    disagreements.to_csv(args.disagreements_output, index=False)
    spot_check = (
        merged[merged["final_label"].notna()]
        .sample(n=min(50, int(merged["final_label"].notna().sum())), random_state=args.seed)
        .copy()
    )
    if not spot_check.empty:
        spot_check["author_label"] = ""
        spot_check["author_matches_consensus"] = ""
        spot_check.to_csv(args.spot_check_output, index=False)

    merged.to_csv(args.merged_output, index=False)
    dataset.to_csv(args.dataset_output, index=False)
    metrics = build_stage4_metrics(metrics, merged, threshold_used, threshold_fallback_triggered)
    with open(args.metrics_output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    if metrics["kappa_status"] != "proceed" and not args.allow_low_kappa:
        raise RuntimeError(
            f"Kappa gate failed with status={metrics['kappa_status']}. "
            f"See {args.metrics_output} and {args.disagreements_output}."
        )
    logger.info("Wrote %s, %s, and %s", args.merged_output, args.dataset_output, args.metrics_output)


if __name__ == "__main__":
    main()
