"""Prepare one or more article corpora for sentiment annotation and training."""

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
    build_input_text,
    ensure_parent_dir,
    normalize_label,
    normalize_text,
    validate_required_columns,
)
from src.utils.io import read_table

logger = logging.getLogger(__name__)

CAFEF_REQUIRED_COLUMNS = ["url", "category", "title", "body_clean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a normalized article-level sentiment training corpus."
    )
    parser.add_argument("--input-file")
    parser.add_argument("--cafef-input")
    parser.add_argument("--extra-input")
    parser.add_argument("--extra-source-name", default="extra")
    parser.add_argument("--extra-date-column", default="time")
    parser.add_argument("--extra-title-column", default="title")
    parser.add_argument("--extra-body-column", default="content")
    parser.add_argument("--extra-category-column", default="category")
    parser.add_argument("--extra-url-column", default="url")
    parser.add_argument("--max-date")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--report-file")
    parser.add_argument("--max-body-chars", type=int, default=1200)
    return parser.parse_args()


def _normalize_published_at(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S").fillna(
        series.astype(str).map(normalize_text)
    )


def prepare_training_dataframe(
    df: pd.DataFrame, *, max_body_chars: int = 1200
) -> pd.DataFrame:
    base_required = [
        "article_id",
        "source",
        "category",
        "published_at",
        "title",
        "body_text",
    ]
    validate_required_columns(df, base_required, dataset_name="training input")

    # Slice long bodies BEFORE normalization to avoid work on discarded characters
    body_series = df["body_text"].fillna("").astype(str).str.slice(0, max_body_chars)

    prepared = pd.DataFrame(
        {
            "article_id": df["article_id"].astype(str),
            "source": df["source"].astype(str).map(normalize_text),
            "category": df["category"].fillna("").astype(str).map(normalize_text),
            "published_at": _normalize_published_at(df["published_at"]),
            "title": df["title"].fillna("").astype(str).map(normalize_text),
            "body_text": body_series.map(normalize_text),
        }
    )
    if "source_dataset" in df.columns:
        prepared["source_dataset"] = (
            df["source_dataset"].astype(str).map(normalize_text)
        )
    # body_text already sliced above; build the input text now
    prepared["input_text"] = [
        build_input_text(title, body_text)
        for title, body_text in zip(prepared["title"], prepared["body_text"])
    ]
    if prepared["article_id"].duplicated().any():
        duplicates = (
            prepared.loc[prepared["article_id"].duplicated(), "article_id"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Training input contains duplicate article_id values: {duplicates[:10]}"
        )

    if {"label", "split"} & set(df.columns):
        if "label" not in df.columns:
            raise ValueError(
                "Training input includes split values but is missing label."
            )
        prepared["label"] = df["label"].astype(str).map(normalize_label)
        if "split" in df.columns:
            prepared["split"] = df["split"].astype(str).str.strip().str.lower()
        else:
            prepared["split"] = assign_splits(prepared, seed=42)
        validate_required_columns(
            prepared, LABELED_REQUIRED_COLUMNS, dataset_name="prepared labeled corpus"
        )
    else:
        validate_required_columns(
            prepared, TRAINING_REQUIRED_COLUMNS, dataset_name="prepared training corpus"
        )

    ordered_columns = TRAINING_REQUIRED_COLUMNS + [
        column for column in prepared.columns if column not in TRAINING_REQUIRED_COLUMNS
    ]
    if "label" in prepared.columns:
        ordered_columns = LABELED_REQUIRED_COLUMNS + [
            column
            for column in prepared.columns
            if column not in LABELED_REQUIRED_COLUMNS
        ]
    return prepared[ordered_columns]


def normalize_cafef_training_corpus(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        df, CAFEF_REQUIRED_COLUMNS, dataset_name="CafeF training corpus"
    )
    published_col = "published_at" if "published_at" in df.columns else "date"
    if published_col not in df.columns:
        raise ValueError("CafeF training corpus requires either published_at or date.")
    return pd.DataFrame(
        {
            "article_id": df["url"].astype(str),
            "source": "cafef",
            "source_dataset": "cafef",
            "category": df["category"].fillna("").astype(str),
            "published_at": df[published_col].astype(str),
            "title": df["title"].fillna("").astype(str),
            "body_text": df["body_clean"].fillna("").astype(str),
        }
    )


def normalize_extra_training_corpus(
    df: pd.DataFrame,
    *,
    source_name: str,
    date_column: str,
    title_column: str,
    body_column: str,
    category_column: str,
    url_column: str,
    max_date: str | None,
) -> pd.DataFrame:
    validate_required_columns(
        df,
        [url_column, title_column, body_column, category_column, date_column],
        dataset_name=f"{source_name} training corpus",
    )
    working = df.copy()
    if max_date:
        parsed = pd.to_datetime(working[date_column], errors="coerce")
        cutoff = pd.Timestamp(max_date)
        working = working.loc[parsed.notna() & (parsed <= cutoff)].copy()
    return pd.DataFrame(
        {
            "article_id": working[url_column].astype(str),
            "source": normalize_text(source_name),
            "source_dataset": normalize_text(source_name),
            "category": working[category_column].fillna("").astype(str),
            "published_at": working[date_column].astype(str),
            "title": working[title_column].fillna("").astype(str),
            "body_text": working[body_column].fillna("").astype(str),
        }
    )


def combine_training_sources(
    cafef_df: pd.DataFrame,
    *,
    extra_df: pd.DataFrame | None = None,
    extra_source_name: str = "extra",
    extra_date_column: str = "time",
    extra_title_column: str = "title",
    extra_body_column: str = "content",
    extra_category_column: str = "category",
    extra_url_column: str = "url",
    max_date: str | None = None,
    max_body_chars: int = 1200,
) -> pd.DataFrame:
    frames = [normalize_cafef_training_corpus(cafef_df)]
    if extra_df is not None:
        frames.append(
            normalize_extra_training_corpus(
                extra_df,
                source_name=extra_source_name,
                date_column=extra_date_column,
                title_column=extra_title_column,
                body_column=extra_body_column,
                category_column=extra_category_column,
                url_column=extra_url_column,
                max_date=max_date,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    if combined["article_id"].duplicated().any():
        duplicates = (
            combined.loc[combined["article_id"].duplicated(), "article_id"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Combined training corpus contains duplicate article_id values across inputs: {duplicates[:10]}"
        )
    return prepare_training_dataframe(combined, max_body_chars=max_body_chars)


def build_report(prepared: pd.DataFrame) -> dict:
    report = {
        "rows": int(len(prepared)),
        "sources": prepared["source"].value_counts().to_dict(),
        "categories": prepared["category"].value_counts().head(20).to_dict(),
        "has_labels": bool("label" in prepared.columns),
    }
    if "source_dataset" in prepared.columns:
        report["source_datasets"] = prepared["source_dataset"].value_counts().to_dict()
        years = pd.to_datetime(
            prepared["published_at"], errors="coerce"
        ).dt.year.astype("Int64")
        report["source_dataset_years"] = pd.crosstab(
            prepared["source_dataset"], years.fillna(-1).astype(int).astype(str)
        ).to_dict(orient="index")
    if "label" in prepared.columns:
        report["labels"] = prepared["label"].value_counts().to_dict()
    if "split" in prepared.columns:
        report["splits"] = prepared["split"].value_counts().to_dict()
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    if args.input_file:
        df = read_table(args.input_file)
        prepared = prepare_training_dataframe(
            df,
            max_body_chars=args.max_body_chars,
        )
    else:
        if not args.cafef_input:
            raise ValueError("Provide either --input-file or --cafef-input.")
        cafef_df = read_table(args.cafef_input)
        extra_df = read_table(args.extra_input) if args.extra_input else None
        prepared = combine_training_sources(
            cafef_df,
            extra_df=extra_df,
            extra_source_name=args.extra_source_name,
            extra_date_column=args.extra_date_column,
            extra_title_column=args.extra_title_column,
            extra_body_column=args.extra_body_column,
            extra_category_column=args.extra_category_column,
            extra_url_column=args.extra_url_column,
            max_date=args.max_date,
            max_body_chars=args.max_body_chars,
        )
    output_path = ensure_parent_dir(args.output_file)
    prepared.to_parquet(output_path, index=False)

    report_file = args.report_file or str(Path(output_path).with_suffix(".report.json"))
    Path(report_file).write_text(
        json.dumps(build_report(prepared), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Prepared %d training rows -> %s", len(prepared), output_path)


if __name__ == "__main__":
    main()
