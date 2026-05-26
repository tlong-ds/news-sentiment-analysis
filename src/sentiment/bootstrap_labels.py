"""Bootstrap market-impact sentiment labels with local or hosted LLM backends."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
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

BATCH_PROMPT_TEMPLATE = """You are labeling Vietnamese business news for investor market impact sentiment.
Classify each article by expected market impact, not writing tone.

Return strict JSON as an array. Each item must include:
- article_id
- label: exactly one of negative, neutral, positive
- confidence: number from 0 to 1
- rationale: short explanation

Articles:
{articles}
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
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
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


def _extract_json_array_candidate(raw_text: str) -> list[dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response does not contain valid JSON array.")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, list):
        raise ValueError("Model response must be a JSON array.")
    return payload


def parse_bootstrap_response(raw_text: str) -> dict[str, Any]:
    payload = _extract_json_candidate(raw_text)
    return parse_bootstrap_item(payload)


def parse_bootstrap_item(payload: dict[str, Any]) -> dict[str, Any]:
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


def build_batch_prompt(rows: list[pd.Series]) -> str:
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            "\n".join(
                [
                    f"article_id: {row.get('article_id', '')}",
                    f"title: {row.get('title', '')}",
                    f"category: {row.get('category', '')}",
                    f"published_at: {row.get('published_at', '')}",
                    f"body: {row.get('body_text', '')}",
                ]
            )
        )
    articles_block = "\n\n".join(formatted_rows)
    return BATCH_PROMPT_TEMPLATE.format(articles=articles_block)


_session = None


def _get_session() -> requests.Session:
    """Return a requests.Session configured with a connection pool suitable for concurrency.

    The pool size can be tuned via the HTTP_POOL_SIZE environment variable.
    """
    global _session
    if _session is None:
        _session = requests.Session()
        # Configure connection pooling for concurrent threads to reuse connections.
        try:
            pool_size = int(os.environ.get("HTTP_POOL_SIZE", "50"))
        except Exception:
            pool_size = 50
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size, pool_maxsize=pool_size
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


def _call_ollama(prompt: str, model_name: str) -> str:
    max_retries = 3
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            response = _get_session().post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("response", "")).strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2.0


def _call_gemini(prompt: str, model_name: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini bootstrap fallback.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    max_retries = 5
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            response = _get_session().post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json"},
                },
                timeout=180,
            )
            if response.status_code == 429:
                if attempt == max_retries - 1:
                    response.raise_for_status()
                time.sleep(backoff)
                backoff *= 2.0
                continue
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
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2.0


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
    concurrency: int = 5,
    batch_size: int = 20,
    sleep_seconds: float = 0.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    validate_required_columns(
        df,
        {"article_id", "source", "category", "published_at", "title", "body_text"},
        dataset_name="bootstrap input",
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    rows: list[dict[str, Any]] = [None] * len(df)
    raw_records: list[dict[str, Any]] = [None] * len(df)

    def process_batch(
        batch_index: int, batch_rows: list[pd.Series]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
        prompt = build_batch_prompt(batch_rows)
        last_error = None
        for spec in model_specs:
            raw_text = ""
            try:
                raw_text = str(model_runner(prompt, spec))
                payloads = _extract_json_array_candidate(raw_text)
                labeled = []
                raw = []
                payload_map = {}
                for payload in payloads:
                    if not isinstance(payload, dict):
                        raise ValueError("Batch response items must be JSON objects.")
                    article_id = str(payload.get("article_id", "")).strip()
                    if not article_id:
                        raise ValueError("Batch response item missing article_id.")
                    if article_id in payload_map:
                        raise ValueError(
                            f"Duplicate article_id in batch response: {article_id}"
                        )
                    payload_map[article_id] = payload
                for row in batch_rows:
                    article_id = str(row["article_id"])
                    if article_id not in payload_map:
                        raise ValueError(
                            f"Batch response missing article_id={article_id}."
                        )
                    parsed = parse_bootstrap_item(payload_map[article_id])
                    labeled.append(
                        {
                            "article_id": article_id,
                            "label": parsed["label"],
                            "confidence": parsed["confidence"],
                            "rationale": parsed["rationale"],
                            "model_name": spec.model_name,
                            "prompt_version": prompt_version,
                        }
                    )
                    raw.append(
                        {
                            "article_id": article_id,
                            "backend": spec.backend,
                            "model_name": spec.model_name,
                            "prompt_version": prompt_version,
                            "response_text": raw_text,
                            "status": "ok",
                            "batch_index": batch_index,
                        }
                    )
                return labeled, raw, spec.backend, spec.model_name
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"Bootstrap labeling failed for batch {batch_index}: {last_error}"
        ) from last_error

    from concurrent.futures import ThreadPoolExecutor

    batches: list[list[tuple[int, pd.Series]]] = []
    current: list[tuple[int, pd.Series]] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        current.append((pos, row))
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    def process_and_track(batch_info):
        batch_index, batch_rows = batch_info
        return batch_index, process_batch(batch_index, batch_rows)

    tasks = [
        (idx, [row for _, row in batch]) for idx, batch in enumerate(batches, start=1)
    ]

    if concurrency > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(process_and_track, tasks))
    else:
        results = [process_and_track(task) for task in tasks]

    for batch_index, (labeled_rows, raw_rows, used_backend, _) in results:
        batch = batches[batch_index - 1]
        indices = [pos for pos, _ in batch]
        for pos, labeled_row, raw_row in zip(indices, labeled_rows, raw_rows):
            rows[pos] = labeled_row
            raw_records[pos] = raw_row
        if used_backend == "gemini" and sleep_seconds > 0:
            time.sleep(sleep_seconds)

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
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
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
