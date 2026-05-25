"""Bootstrap market-impact sentiment labels with local or hosted LLM backends."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.sentiment.common import (
    DEFAULT_PROMPT_VERSION,
    ensure_parent_dir,
    normalize_label,
    validate_required_columns,
)
from src.utils.io import read_parquet_table

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are labeling Vietnamese business news for investor market impact sentiment.
Classify the article by expected market impact, not writing tone.

Return strict JSON with keys:
- label: exactly one of negative, neutral, positive
- confidence: number from 0 to 1
- rationale: short explanation

Title: {title}
Category: {category}
Published at: {published_at}
Body: {body_text}
"""


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    model_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap sentiment labels with LLM backends."
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--raw-output-file")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--fallback-models", nargs="*", default=["nemotron-3-nano:4b"])
    parser.add_argument(
        "--gemini-fallback-models",
        nargs="*",
        default=["gemini-2.0-flash-lite", "gemini-2.0-flash"],
    )
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--report-file")
    return parser.parse_args()


def build_prompt(row: pd.Series) -> str:
    return PROMPT_TEMPLATE.format(
        title=row.get("title", ""),
        category=row.get("category", ""),
        published_at=row.get("published_at", ""),
        body_text=row.get("body_text", ""),
    )


def _extract_json_candidate(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response does not contain valid JSON.")
        return json.loads(text[start : end + 1])


def parse_bootstrap_response(raw_text: str) -> dict[str, Any]:
    payload = _extract_json_candidate(raw_text)
    label = normalize_label(str(payload.get("label", "")))
    confidence = float(payload.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}.")
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("Bootstrap response rationale is required.")
    return {
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
    }


def _call_ollama(prompt: str, model_name: str) -> str:
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False, "format": "json"},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("response", "")).strip()


def _call_gemini(prompt: str, model_name: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini bootstrap fallback.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    response = requests.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise ValueError("Gemini returned an empty text response.")
    return text


def call_model(prompt: str, spec: ModelSpec) -> str:
    if spec.backend == "ollama":
        return _call_ollama(prompt, spec.model_name)
    if spec.backend == "gemini":
        return _call_gemini(prompt, spec.model_name)
    raise ValueError(f"Unsupported bootstrap backend: {spec.backend}")


def default_model_specs(
    primary_backend: str,
    primary_model: str,
    fallback_models: list[str],
    gemini_models: list[str],
) -> list[ModelSpec]:
    specs = [ModelSpec(primary_backend, primary_model)]
    specs.extend(
        ModelSpec(primary_backend, model_name) for model_name in fallback_models
    )
    specs.extend(ModelSpec("gemini", model_name) for model_name in gemini_models)
    return specs


def bootstrap_labels_frame(
    df: pd.DataFrame,
    *,
    model_specs: list[ModelSpec],
    prompt_version: str,
    model_runner: Any = call_model,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    validate_required_columns(
        df,
        {"article_id", "source", "category", "published_at", "title", "body_text"},
        dataset_name="bootstrap input",
    )
    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        prompt = build_prompt(row)
        last_error = None
        for spec in model_specs:
            raw_text = ""
            try:
                raw_text = str(model_runner(prompt, spec))
                parsed = parse_bootstrap_response(raw_text)
                rows.append(
                    {
                        "article_id": str(row["article_id"]),
                        "label": parsed["label"],
                        "confidence": parsed["confidence"],
                        "rationale": parsed["rationale"],
                        "model_name": spec.model_name,
                        "prompt_version": prompt_version,
                    }
                )
                raw_records.append(
                    {
                        "article_id": str(row["article_id"]),
                        "backend": spec.backend,
                        "model_name": spec.model_name,
                        "prompt_version": prompt_version,
                        "response_text": raw_text,
                        "status": "ok",
                    }
                )
                break
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised through fallback behavior
                last_error = exc
                raw_records.append(
                    {
                        "article_id": str(row["article_id"]),
                        "backend": spec.backend,
                        "model_name": spec.model_name,
                        "prompt_version": prompt_version,
                        "response_text": raw_text,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        else:
            raise RuntimeError(
                f"Bootstrap labeling failed for article_id={row['article_id']}: {last_error}"
            ) from last_error
    return pd.DataFrame(rows), raw_records


def build_report(
    labels_df: pd.DataFrame, *, confidence_threshold: float
) -> dict[str, Any]:
    report = {
        "rows": int(len(labels_df)),
        "labels": labels_df["label"].value_counts().to_dict(),
        "models": labels_df["model_name"].value_counts().to_dict(),
        "prompt_versions": labels_df["prompt_version"].value_counts().to_dict(),
        "confidence_threshold": confidence_threshold,
        "auto_accept_rows": int(
            (labels_df["confidence"] >= confidence_threshold).sum()
        ),
    }
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    input_df = read_parquet_table(args.input_file)
    if args.limit:
        input_df = input_df.head(args.limit).copy()
    specs = default_model_specs(
        args.backend, args.model, args.fallback_models, args.gemini_fallback_models
    )
    labels_df, raw_records = bootstrap_labels_frame(
        input_df,
        model_specs=specs,
        prompt_version=args.prompt_version,
    )
    output_path = ensure_parent_dir(args.output_file)
    labels_df.to_parquet(output_path, index=False)

    raw_output_path = args.raw_output_file or str(
        Path(output_path).with_suffix(".raw.jsonl")
    )
    raw_target = ensure_parent_dir(raw_output_path)
    with raw_target.open("w", encoding="utf-8") as handle:
        for record in raw_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_file = args.report_file or str(Path(output_path).with_suffix(".report.json"))
    Path(report_file).write_text(
        json.dumps(
            build_report(labels_df, confidence_threshold=args.confidence_threshold),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Bootstrapped %d labels -> %s", len(labels_df), output_path)


if __name__ == "__main__":
    main()
