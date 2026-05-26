"""Shared helpers for the supervised sentiment pipeline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoConfig

try:
    import underthesea
except ImportError:  # pragma: no cover - fallback used when dependency is absent
    underthesea = None

try:
    from vncorenlp import VnCoreNLP
except ImportError:
    VnCoreNLP = None

from src.config import VNCORENLP_JAR_PATH

_vncorenlp_annotator = None


def get_vncorenlp_annotator() -> VnCoreNLP | None:
    global _vncorenlp_annotator
    if _vncorenlp_annotator is not None:
        return _vncorenlp_annotator
    if VnCoreNLP is None:
        return None

    import contextlib
    import logging
    import os
    import subprocess

    def _ensure_java_on_path() -> None:
        try:
            subprocess.check_call(
                ["java", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            return

        conda_java_bin = os.path.join(conda_prefix, "lib", "jvm", "bin")
        if not os.path.exists(os.path.join(conda_java_bin, "java")):
            return

        os.environ["PATH"] = conda_java_bin + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("JAVA_HOME", os.path.join(conda_prefix, "lib", "jvm"))

    jar_path = os.path.abspath(VNCORENLP_JAR_PATH)
    if not os.path.exists(jar_path):
        return None

    _ensure_java_on_path()

    jar_dir = os.path.dirname(jar_path)
    jar_name = os.path.basename(jar_path)

    try:
        cwd = os.getcwd()
        with contextlib.ExitStack() as stack:
            stack.callback(lambda: os.chdir(cwd))
            os.chdir(jar_dir)
            _vncorenlp_annotator = VnCoreNLP(jar_name, annotators="wseg")
        return _vncorenlp_annotator
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Failed to initialize VnCoreNLP, falling back to underthesea: %s", e
        )
        return None


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
    "input_text_segmented",
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
    input_text_segmented: str


def segment_text(text: str) -> str:
    """Word-segment text for PhoBERT, falling back to underthesea or whitespace normalization."""
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""

    annotator = get_vncorenlp_annotator()
    if annotator is not None:
        try:
            sentences = annotator.tokenize(normalized)
            words = []
            for sentence in sentences:
                words.extend(sentence)
            return " ".join(words)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "VnCoreNLP tokenization failed, falling back: %s", e
            )

    if underthesea is None:
        return normalized
    return underthesea.word_tokenize(normalized, format="text")


def normalize_presegmented_text(text: str) -> str:
    """Normalize already segmented text while preserving PhoBERT-style underscores."""
    return " ".join(str(text or "").split())


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
    label_col: str = "label",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> pd.Series:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Split ratios must satisfy 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1."
        )
    if split_col in df.columns and df[split_col].notna().all():
        normalized = df[split_col].astype(str).str.strip().str.lower()
        invalid = sorted(set(normalized) - VALID_SPLITS)
        if invalid:
            raise ValueError(
                f"Invalid split values: {invalid}. Expected one of {sorted(VALID_SPLITS)}"
            )
        return normalized

    rng = random.Random(seed)
    assignments = pd.Series(index=df.index, dtype="object")
    group_source = (
        df[label_col].astype(str)
        if label_col in df.columns
        else pd.Series(["all"] * len(df), index=df.index)
    )
    for _, group_idx in group_source.groupby(group_source).groups.items():
        indices = list(group_idx)
        rng.shuffle(indices)
        total = len(indices)
        train_cut = (
            max(1, int(round(total * train_ratio))) if total >= 3 else max(1, total - 1)
        )
        val_cut = max(1, int(round(total * val_ratio))) if total >= 10 else 1
        if train_cut + val_cut >= total:
            val_cut = 1 if total >= 3 else 0
            train_cut = max(1, total - val_cut - 1) if total >= 3 else total - val_cut
        for offset, idx in enumerate(indices):
            if offset < train_cut:
                assignments.loc[idx] = "train"
            elif offset < train_cut + val_cut:
                assignments.loc[idx] = "val"
            else:
                assignments.loc[idx] = "test"
    assignments = assignments.fillna("train")
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
