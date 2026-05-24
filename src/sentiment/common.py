"""Shared helpers for the supervised sentiment pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import underthesea
except ImportError:  # pragma: no cover - fallback used when dependency is absent
    underthesea = None


SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

DEFAULT_PROMPT_VERSION = "v1_market_interpretation"


@dataclass(frozen=True)
class PreparedInputRow:
    article_id: str
    source: str
    category: str
    date: str
    title: str
    body_lead: str
    input_text: str
    input_text_segmented: str
    token_count: int


def segment_text(text: str) -> str:
    """Word-segment text for PhoBERT, falling back to whitespace normalization."""
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    if underthesea is None:
        return normalized
    return underthesea.word_tokenize(normalized, format="text")


def normalize_presegmented_text(text: str) -> str:
    """Normalize already segmented text while preserving PhoBERT-style underscores."""
    return " ".join(str(text or "").split())


def body_lead(text: str, max_chars: int = 300) -> str:
    """Return the first N characters of a cleaned body string."""
    collapsed = " ".join(str(text or "").split())
    return collapsed[:max_chars].strip()


def build_input_text(title: str, lead: str) -> str:
    title_clean = " ".join(str(title or "").split()).strip()
    lead_clean = " ".join(str(lead or "").split()).strip()
    if title_clean and lead_clean:
        return f"{title_clean} . {lead_clean}"
    return title_clean or lead_clean


def count_tokens(segmented_text: str) -> int:
    return len([token for token in segmented_text.split() if token])


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    return bucket_ids.map(lambda value: f"period_{value + 1}" if value < buckets else f"period_{buckets}")


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
