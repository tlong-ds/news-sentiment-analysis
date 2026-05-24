from pathlib import Path

import pandas as pd

from src.sentiment.prepare_inputs import prepare_cafef_inputs, prepare_vific_inputs
from src.sentiment.sample_vific import stratified_sample


def test_prepare_cafef_inputs_builds_expected_columns(tmp_path: Path):
    articles = pd.DataFrame(
        [
            {
                "url": "u1",
                "category": "Chứng khoán",
                "date": "2024-01-02",
                "trading_date": "2024-01-02",
                "title": "Lợi nhuận tăng mạnh",
                "body_clean": "Doanh nghiệp ghi nhận lợi nhuận tăng trưởng mạnh trong quý này." * 4,
            }
        ]
    )
    input_path = tmp_path / "articles_clean.csv"
    output_path = tmp_path / "cafef_input.csv"
    articles.to_csv(input_path, index=False)

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
        "input_text_segmented",
        "token_count",
    ]
    assert prepared.loc[0, "article_id"] == "u1"
    assert prepared.loc[0, "body_lead"]
    assert prepared.loc[0, "token_count"] >= 5


def test_prepare_vific_inputs_filters_short_rows(tmp_path: Path):
    rows = pd.DataFrame(
        [
            {"article_id": "a1", "date": "2024-01-01", "title": "T1", "body": "ngắn"},
            {"article_id": "a2", "date": "2024-01-02", "title": "Tin tốt", "body": "nội dung dài hơn " * 20},
        ]
    )
    input_path = tmp_path / "vific.csv"
    output_path = tmp_path / "vific_input.csv"
    rows.to_csv(input_path, index=False)

    prepared = prepare_vific_inputs(input_path, output_path)

    assert prepared["article_id"].tolist() == ["a2"]


def test_prepare_vific_inputs_preserves_existing_underscores(tmp_path: Path):
    rows = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "date": "2024-01-02",
                "title": "thị_trường chứng_khoán",
                "body": "nhà_đầu_tư kỳ_vọng thanh_khoản cải_thiện " * 10,
            }
        ]
    )
    input_path = tmp_path / "vific.csv"
    output_path = tmp_path / "vific_input.csv"
    rows.to_csv(input_path, index=False)

    prepared = prepare_vific_inputs(input_path, output_path, preserve_existing_segmentation=True)

    assert "thị_trường" in prepared.loc[0, "input_text_segmented"]
    assert "nhà_đầu_tư" in prepared.loc[0, "input_text_segmented"]


def test_stratified_sample_is_deterministic():
    df = pd.DataFrame(
        [
            {
                "article_id": f"id{i}",
                "category": "A" if i % 2 == 0 else "B",
                "date": f"2024-01-{(i % 9) + 1:02d}",
                "token_count": 20 + i,
            }
            for i in range(20)
        ]
    )

    first = stratified_sample(df, sample_size=8, seed=7)
    second = stratified_sample(df, sample_size=8, seed=7)
    assert first["article_id"].tolist() == second["article_id"].tolist()
