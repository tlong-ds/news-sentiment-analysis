from pathlib import Path

import pandas as pd
import pytest

from src.sentiment.prepare_inputs import prepare_cafef_inputs


def test_prepare_cafef_inputs_builds_expected_columns(tmp_path: Path):
    articles = pd.DataFrame(
        [
            {
                "url": "u1",
                "category": "Chứng khoán",
                "date": "2024-01-02",
                "trading_date": "2024-01-02",
                "title": "Lợi nhuận tăng mạnh",
                "body_clean": "Doanh nghiệp ghi nhận lợi nhuận tăng trưởng mạnh trong quý này."
                * 4,
            }
        ]
    )
    input_path = tmp_path / "articles_clean.parquet"
    output_path = tmp_path / "cafef_input.parquet"
    articles.to_parquet(input_path, index=False)

    prepared = prepare_cafef_inputs(input_path, output_path)

    assert list(prepared.columns) == [
        "article_id",
        "url",
        "trading_date",
        "source",
        "category",
        "date",
        "title",
        "body_lead",
        "input_text",
        "token_count",
    ]
    assert prepared.loc[0, "article_id"] == "u1"
    assert prepared.loc[0, "body_lead"]
    assert prepared.loc[0, "token_count"] >= 5


def test_prepare_cafef_inputs_requires_parquet_input(tmp_path: Path):
    articles = pd.DataFrame(
        [
            {
                "url": "u1",
                "category": "Chứng khoán",
                "date": "2024-01-02",
                "trading_date": "2024-01-02",
                "title": "Lợi nhuận tăng mạnh",
                "body_clean": "Doanh nghiệp ghi nhận lợi nhuận tăng trưởng mạnh trong quý này."
                * 4,
            }
        ]
    )
    input_path = tmp_path / "articles_clean.csv"
    output_path = tmp_path / "cafef_input.parquet"
    articles.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Expected a parquet artifact"):
        prepare_cafef_inputs(input_path, output_path)
