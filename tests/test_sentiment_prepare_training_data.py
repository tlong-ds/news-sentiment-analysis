import pandas as pd
import pytest

from src.sentiment.prepare_training_data import (
    combine_training_sources,
    normalize_extra_training_corpus,
    prepare_training_dataframe,
)


def _cafef_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "url": "u1",
                "category": "Chứng khoán",
                "published_at": "2024-01-02T09:00:00",
                "title": "CafeF row",
                "body_clean": "noi dung cafef",
            }
        ]
    )


def _full_data_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "url": "f1",
                "category": "Tài chính",
                "time": "2024-01-03 10:00:00",
                "title": "Full data kept",
                "content": "noi dung full data duoc giu lai",
            },
            {
                "url": "f2",
                "category": "Tài chính",
                "time": "2025-01-03 10:00:00",
                "title": "Full data trimmed",
                "content": "dong nay phai bi cat",
            },
        ]
    )


def test_combine_training_sources_trims_and_maps_extra_dataset():
    prepared = combine_training_sources(
        _cafef_df(),
        extra_df=_full_data_df(),
        extra_source_name="full_data",
        extra_date_column="time",
        extra_title_column="title",
        extra_body_column="content",
        extra_category_column="category",
        extra_url_column="url",
        max_date="2024-12-31",
    )

    assert set(prepared["source_dataset"]) == {"cafef", "full_data"}
    assert "f2" not in set(prepared["article_id"])
    full_data_row = prepared.loc[prepared["article_id"] == "f1"].iloc[0]
    assert full_data_row["source"] == "full_data"
    assert full_data_row["body_text"] == "noi dung full data duoc giu lai"
    assert full_data_row["published_at"].startswith("2024-01-03")


def test_prepare_training_dataframe_supports_extra_only():
    normalized = normalize_extra_training_corpus(
        _full_data_df(),
        source_name="full_data",
        date_column="time",
        title_column="title",
        body_column="content",
        category_column="category",
        url_column="url",
        max_date="2024-12-31",
    )
    prepared = prepare_training_dataframe(normalized)

    assert set(prepared["source_dataset"]) == {"full_data"}
    assert set(prepared["article_id"]) == {"f1"}
    assert "input_text" in prepared.columns


def test_combine_training_sources_rejects_missing_columns_and_duplicate_overlap():
    with pytest.raises(ValueError, match="missing required columns"):
        combine_training_sources(
            _cafef_df(),
            extra_df=pd.DataFrame([{"url": "f1", "title": "x"}]),
            extra_source_name="full_data",
            max_date="2024-12-31",
        )

    with pytest.raises(ValueError, match="duplicate article_id"):
        combine_training_sources(
            _cafef_df(),
            extra_df=pd.DataFrame(
                [
                    {
                        "url": "u1",
                        "category": "Tài chính",
                        "time": "2024-01-03 10:00:00",
                        "title": "duplicate",
                        "content": "duplicate row",
                    }
                ]
            ),
            extra_source_name="full_data",
            max_date="2024-12-31",
        )
