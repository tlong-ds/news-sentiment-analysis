import pandas as pd

from src.sentiment.build_silver_labels import (
    build_consensus_table,
    parse_response_records,
    split_labeled_dataset,
)


def test_parse_response_records_reads_json_payload():
    records = [
        {
            "article_id": "a1",
            "model": "m1",
            "timestamp": "2026-01-01T00:00:00Z",
            "response_text": '{"label":"positive","confidence":0.9,"reason":"tot"}',
        }
    ]
    df = parse_response_records(records, "llm_a")
    assert df.loc[0, "llm_a_label"] == "positive"
    assert df.loc[0, "llm_a_confidence"] == 0.9


def test_build_consensus_table_and_split():
    sample_df = pd.DataFrame(
        [
            {"article_id": "a1", "input_text": "x", "input_text_segmented": "x x x x x"},
            {"article_id": "a2", "input_text": "y", "input_text_segmented": "y y y y y"},
            {"article_id": "a3", "input_text": "z", "input_text_segmented": "z z z z z"},
            {"article_id": "a4", "input_text": "w", "input_text_segmented": "w w w w w"},
            {"article_id": "a5", "input_text": "n", "input_text_segmented": "n n n n n"},
            {"article_id": "a6", "input_text": "m", "input_text_segmented": "m m m m m"},
        ]
    )
    llm_a_df = pd.DataFrame(
        [
            {"article_id": "a1", "llm_a_label": "positive", "llm_a_confidence": 0.9, "llm_a_reason": "", "llm_a_model": "a", "annotation_timestamp": "t"},
            {"article_id": "a2", "llm_a_label": "negative", "llm_a_confidence": 0.9, "llm_a_reason": "", "llm_a_model": "a", "annotation_timestamp": "t"},
            {"article_id": "a3", "llm_a_label": "neutral", "llm_a_confidence": 0.9, "llm_a_reason": "", "llm_a_model": "a", "annotation_timestamp": "t"},
            {"article_id": "a4", "llm_a_label": "positive", "llm_a_confidence": 0.9, "llm_a_reason": "", "llm_a_model": "a", "annotation_timestamp": "t"},
            {"article_id": "a5", "llm_a_label": "negative", "llm_a_confidence": 0.9, "llm_a_reason": "", "llm_a_model": "a", "annotation_timestamp": "t"},
            {"article_id": "a6", "llm_a_label": "neutral", "llm_a_confidence": 0.9, "llm_a_reason": "", "llm_a_model": "a", "annotation_timestamp": "t"},
        ]
    )
    llm_b_df = pd.DataFrame(
        [
            {"article_id": "a1", "llm_b_label": "positive", "llm_b_confidence": 0.95, "llm_b_reason": "", "llm_b_model": "b", "annotation_timestamp": "t"},
            {"article_id": "a2", "llm_b_label": "negative", "llm_b_confidence": 0.95, "llm_b_reason": "", "llm_b_model": "b", "annotation_timestamp": "t"},
            {"article_id": "a3", "llm_b_label": "neutral", "llm_b_confidence": 0.95, "llm_b_reason": "", "llm_b_model": "b", "annotation_timestamp": "t"},
            {"article_id": "a4", "llm_b_label": "positive", "llm_b_confidence": 0.95, "llm_b_reason": "", "llm_b_model": "b", "annotation_timestamp": "t"},
            {"article_id": "a5", "llm_b_label": "negative", "llm_b_confidence": 0.95, "llm_b_reason": "", "llm_b_model": "b", "annotation_timestamp": "t"},
            {"article_id": "a6", "llm_b_label": "neutral", "llm_b_confidence": 0.95, "llm_b_reason": "", "llm_b_model": "b", "annotation_timestamp": "t"},
        ]
    )

    merged, metrics = build_consensus_table(sample_df, llm_a_df, llm_b_df, confidence_threshold=0.75)
    dataset = split_labeled_dataset(merged, seed=42)

    assert metrics["kappa"] == 1.0
    assert merged["final_label"].notna().all()
    assert set(dataset["split"]) == {"train", "val", "test"}
    assert set(dataset["label"]) == {"positive", "negative", "neutral"}
