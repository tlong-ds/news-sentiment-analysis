"""Prepare shared ViFiC and CafeF sentiment inputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import CAFEF_DATA_DIR, FINETUNES_DATA_DIR, PROCESSED_DATA_DIR, VIFIC_NORMALIZED_DIR
from src.sentiment.common import (
    body_lead,
    build_input_text,
    count_tokens,
    normalize_presegmented_text,
    segment_text,
    token_stats,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ViFiC and CafeF sentiment input tables.")
    parser.add_argument("--vific-input", default=None, help="Raw ViFiC CSV/TXT-derived article table.")
    parser.add_argument("--cafef-input", default=f"{PROCESSED_DATA_DIR}/articles_clean.csv")
    parser.add_argument("--vific-output", default=f"{VIFIC_NORMALIZED_DIR}/vific_input.csv")
    parser.add_argument("--vific-pretraining-output", default=f"{VIFIC_NORMALIZED_DIR}/vific_pretraining.csv")
    parser.add_argument("--cafef-output", default=f"{CAFEF_DATA_DIR}/cafef_input.csv")
    parser.add_argument("--report-file", default=f"{VIFIC_NORMALIZED_DIR}/input_preparation_report.json")
    parser.add_argument("--max-body-chars", type=int, default=300)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument(
        "--vific-presegmented",
        action="store_true",
        default=True,
        help="Treat ViFiC text as already word-segmented with underscore tokens.",
    )
    parser.add_argument(
        "--no-vific-presegmented",
        action="store_false",
        dest="vific_presegmented",
        help="Force underthesea segmentation for ViFiC text.",
    )
    return parser.parse_args()


def _default_vific_input() -> str:
    candidates = sorted(Path(FINETUNES_DATA_DIR).rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No ViFiC CSV found under data/fine-tunes/. Pass --vific-input explicitly."
        )
    return str(candidates[0])


def _prepare_frame(
    df: pd.DataFrame,
    *,
    article_id_col: str,
    source_default: str,
    category_col: str | None,
    date_col: str,
    title_col: str,
    body_col: str,
    include_url: bool = False,
    include_trading_date: bool = False,
    preserve_existing_segmentation: bool = False,
    max_body_chars: int = 300,
    min_tokens: int = 5,
    max_tokens: int = 220,
) -> pd.DataFrame:
    prepared = pd.DataFrame(
        {
            "article_id": df[article_id_col].astype(str),
            "source": df["source"].astype(str) if "source" in df.columns else source_default,
            "category": df[category_col].astype(str) if category_col and category_col in df.columns else "",
            "date": df[date_col].astype(str),
            "title": df[title_col].fillna("").astype(str),
        }
    )
    prepared["body_lead"] = df[body_col].fillna("").astype(str).map(
        lambda value: body_lead(value, max_chars=max_body_chars)
    )
    prepared["input_text"] = [
        build_input_text(title, lead)
        for title, lead in zip(prepared["title"], prepared["body_lead"])
    ]
    segmenter = normalize_presegmented_text if preserve_existing_segmentation else segment_text
    prepared["input_text_segmented"] = prepared["input_text"].map(segmenter)
    prepared["token_count"] = prepared["input_text_segmented"].map(count_tokens)

    if include_url:
        prepared["url"] = df["url"].astype(str)
    if include_trading_date:
        prepared["trading_date"] = df["trading_date"].astype(str)

    prepared = prepared[prepared["token_count"].between(min_tokens, max_tokens)].copy()
    return prepared.reset_index(drop=True)


def build_input_report(raw_df: pd.DataFrame, prepared_df: pd.DataFrame, sample_size: int = 1000) -> dict:
    sample = prepared_df.head(sample_size)
    return {
        "raw_rows": int(len(raw_df)),
        "prepared_rows": int(len(prepared_df)),
        "dropped_rows": int(len(raw_df) - len(prepared_df)),
        "token_stats_full": token_stats(prepared_df["token_count"]),
        "token_stats_sample": token_stats(sample["token_count"]),
        "sample_size_for_p95": int(min(sample_size, len(prepared_df))),
        "p95_under_200": bool(token_stats(sample["token_count"])["p95"] < 200.0) if len(sample) else False,
    }


def prepare_vific_inputs(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    preserve_existing_segmentation: bool = True,
    max_body_chars: int = 300,
    min_tokens: int = 5,
    max_tokens: int = 220,
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    article_id_col = "article_id" if "article_id" in df.columns else "url"
    category_col = "category" if "category" in df.columns else None
    date_col = "date" if "date" in df.columns else "publish_date"
    title_col = "title" if "title" in df.columns else "Title"
    if "body" in df.columns:
        body_col = "body"
    elif "content" in df.columns:
        body_col = "content"
    else:
        body_col = "text"

    prepared = _prepare_frame(
        df,
        article_id_col=article_id_col,
        source_default="vific",
        category_col=category_col,
        date_col=date_col,
        title_col=title_col,
        body_col=body_col,
        preserve_existing_segmentation=preserve_existing_segmentation,
        max_body_chars=max_body_chars,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    logger.info("Prepared %d ViFiC rows -> %s", len(prepared), output_path)
    return prepared


def prepare_cafef_inputs(
    articles_clean_path: str | Path,
    output_path: str | Path,
    *,
    max_body_chars: int = 300,
    min_tokens: int = 5,
    max_tokens: int = 220,
) -> pd.DataFrame:
    df = pd.read_csv(articles_clean_path)
    prepared = _prepare_frame(
        df,
        article_id_col="url",
        source_default="cafef",
        category_col="category",
        date_col="date",
        title_col="title",
        body_col="body_clean",
        include_url=True,
        include_trading_date=True,
        max_body_chars=max_body_chars,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    front_cols = [
        "article_id",
        "url",
        "trading_date",
        "source",
        "category",
        "date",
        "title",
        "body_lead",
        "input_text",
        "input_text_segmented",
        "token_count",
    ]
    prepared = prepared[front_cols]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    logger.info("Prepared %d CafeF rows -> %s", len(prepared), output_path)
    return prepared


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    vific_input = args.vific_input or _default_vific_input()
    vific_raw_df = pd.read_csv(vific_input)
    vific_prepared = prepare_vific_inputs(
        vific_input,
        args.vific_output,
        preserve_existing_segmentation=args.vific_presegmented,
        max_body_chars=args.max_body_chars,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    vific_pretraining = vific_prepared[["article_id", "input_text_segmented", "token_count"]].copy()
    Path(args.vific_pretraining_output).parent.mkdir(parents=True, exist_ok=True)
    vific_pretraining.to_csv(args.vific_pretraining_output, index=False)
    cafef_raw_df = pd.read_csv(args.cafef_input)
    cafef_prepared = prepare_cafef_inputs(
        args.cafef_input,
        args.cafef_output,
        max_body_chars=args.max_body_chars,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    report = {
        "vific": build_input_report(vific_raw_df, vific_prepared),
        "cafef": build_input_report(cafef_raw_df, cafef_prepared),
        "max_body_chars": args.max_body_chars,
        "token_filter": {"min_tokens": args.min_tokens, "max_tokens": args.max_tokens},
        "vific_pretraining_output": args.vific_pretraining_output,
    }
    Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_file).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote input preparation report -> %s", args.report_file)


if __name__ == "__main__":
    main()
