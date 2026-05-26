"""Shared helpers for the supervised sentiment pipeline."""

from __future__ import annotations

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoConfig


SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
TRAINING_REQUIRED_COLUMNS = [
    "article_id",
    "source",
    "category",
    "published_at",
    "title",
    "body_text",
    "input_text",
]
LABELED_REQUIRED_COLUMNS = TRAINING_REQUIRED_COLUMNS + ["label", "split"]
INFERENCE_REQUIRED_COLUMNS = [
    "url",
    "trading_date",
    "category",
    "sentiment_score",
    "sentiment_label",
    "prob_positive",
    "prob_negative",
    "prob_neutral",
]
DEFAULT_MODEL_SUBDIR = "phobert-sentiment/latest"
DEFAULT_PROMPT_VERSION = "v1_market_interpretation"
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class PreparedInputRow:
    article_id: str
    source: str
    category: str
    published_at: str
    title: str
    body_text: str
    input_text: str


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def body_lead(text: str, max_chars: int = 300) -> str:
    """Return the first N characters of a cleaned body string."""
    collapsed = normalize_text(text)
    return collapsed[:max_chars].strip()


def build_input_text(title: str, lead: str) -> str:
    title_clean = normalize_text(title)
    lead_clean = normalize_text(lead)
    if title_clean and lead_clean:
        return f"{title_clean} . {lead_clean}"
    return title_clean or lead_clean


def count_tokens(segmented_text: str) -> int:
    return len([token for token in segmented_text.split() if token])


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_period_bucket(series: pd.Series, buckets: int = 4) -> pd.Series:
    """Bucket datelike values into ordinal quantile-style periods."""
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.dropna().sort_values()
    if valid.empty:
        return pd.Series(["unknown"] * len(series), index=series.index)

    ranks = parsed.rank(method="first", pct=True)
    bucket_ids = (ranks.fillna(0).clip(lower=0, upper=0.999999) * buckets).astype(int)
    return bucket_ids.map(
        lambda value: f"period_{value + 1}" if value < buckets else f"period_{buckets}"
    )


def token_stats(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p95": 0.0,
        }
    return {
        "count": int(len(values)),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
    }


def validate_required_columns(
    df: pd.DataFrame, required_columns: list[str] | set[str], *, dataset_name: str
) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


def normalize_label(value: str) -> str:
    normalized = normalize_text(value).lower()
    if normalized not in LABEL_TO_ID:
        raise ValueError(
            f"Invalid sentiment label '{value}'. Expected one of: {', '.join(SENTIMENT_LABELS)}"
        )
    return normalized


def assign_splits(
    df: pd.DataFrame,
    *,
    split_col: str = "split",
    time_col: str = "published_at",
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
) -> pd.Series:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Split ratios must satisfy 0 < train_ratio, 0 <= val_ratio and train_ratio + val_ratio < 1."
        )
    if split_col in df.columns and df[split_col].notna().all():
        normalized = df[split_col].astype(str).str.strip().str.lower()
        invalid = sorted(set(normalized) - VALID_SPLITS)
        if invalid:
            raise ValueError(
                f"Invalid split values: {invalid}. Expected one of {sorted(VALID_SPLITS)}"
            )
        return normalized
    if time_col not in df.columns:
        raise ValueError(f"Temporal split requires column '{time_col}'.")
    parsed = pd.to_datetime(df[time_col], errors="coerce")
    if parsed.isna().any():
        raise ValueError(
            f"Temporal split requires valid {time_col} values for all rows."
        )
    day_series = parsed.dt.normalize()
    unique_days = sorted(day_series.unique())
    total_days = len(unique_days)
    if total_days <= 1:
        assignments = pd.Series(["train"] * len(df), index=df.index)
        expected_splits = ["train", "test"]
        if val_ratio > 0.0:
            expected_splits.append("val")
        if len(df) >= len(expected_splits):
            for idx, split_name in enumerate(expected_splits):
                assignments.iloc[idx] = split_name
        return assignments
    train_days = max(1, int(round(total_days * train_ratio)))
    if train_days >= total_days:
        train_days = total_days - 1
    val_days = max(0, int(round(total_days * val_ratio)))
    if train_days + val_days >= total_days:
        val_days = max(0, total_days - train_days - 1)
    train_set = set(unique_days[:train_days])
    val_set = set(unique_days[train_days : train_days + val_days])
    assignments = pd.Series(index=df.index, dtype="object")
    assignments.loc[day_series.isin(train_set)] = "train"
    if val_set:
        assignments.loc[day_series.isin(val_set)] = "val"
    assignments = assignments.fillna("test")

    # Ensure active splits are non-empty if we have enough rows
    expected_splits = ["train", "test"]
    if val_ratio > 0.0:
        expected_splits.append("val")
    if len(df) >= len(expected_splits):
        for split_name in expected_splits:
            if not (assignments == split_name).any():
                largest_split = assignments.value_counts().index[0]
                idx_to_change = assignments[assignments == largest_split].index[0]
                assignments.loc[idx_to_change] = split_name

    return assignments


def default_model_dir(models_root: str | Path) -> Path:
    return Path(models_root) / DEFAULT_MODEL_SUBDIR


def validate_classifier_checkpoint(model_dir: str | Path) -> dict[str, Any]:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing classifier checkpoint at {model_path}. Run `python -m src.sentiment.train_classifier --output-dir {model_path}` first."
        )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint missing config.json at {model_path}. Re-run training to create a valid sequence-classification checkpoint."
        )

    config = AutoConfig.from_pretrained(str(model_path))
    architectures = [value.lower() for value in getattr(config, "architectures", [])]
    if not any("sequenceclassification" in value for value in architectures):
        raise ValueError(
            f"Checkpoint at {model_path} is not a sequence-classification model. Expected a trained classifier, not an MLM or base encoder."
        )
    if getattr(config, "num_labels", None) != 3:
        raise ValueError(
            f"Checkpoint at {model_path} exposes num_labels={getattr(config, 'num_labels', None)}. Expected num_labels=3."
        )
    raw_id2label = getattr(config, "id2label", {}) or {}
    id2label = {int(key): str(value) for key, value in raw_id2label.items()}
    missing_ids = [
        idx for idx in range(3) if id2label.get(idx, "").lower() != ID_TO_LABEL[idx]
    ]
    if missing_ids:
        raise ValueError(
            f"Checkpoint at {model_path} has an invalid id2label mapping. Expected {ID_TO_LABEL}."
        )
    return {
        "model_dir": str(model_path),
        "architectures": getattr(config, "architectures", []),
        "num_labels": int(config.num_labels),
        "id2label": {str(key): value for key, value in id2label.items()},
    }
