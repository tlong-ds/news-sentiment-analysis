import pandas as pd

from src.sentiment.validate_inference import validate_outputs


def test_validate_outputs_reports_probability_drift():
    articles_df = pd.DataFrame({"url": ["u1", "u2"]})
    sentiment_df = pd.DataFrame(
        {
            "url": ["u1", "u2"],
            "trading_date": ["2024-01-02", "2024-01-03"],
            "category": ["Vĩ mô", "Chứng khoán"],
            "sentiment_score": [0.2, -0.3],
            "sentiment_label": ["positive", "negative"],
            "prob_positive": [0.6, 0.1],
            "prob_negative": [0.2, 0.7],
            "prob_neutral": [0.2, 0.2],
        }
    )

    diagnostics = validate_outputs(articles_df, sentiment_df)

    assert diagnostics["row_count_match"] is True
    assert diagnostics["duplicate_urls"] == 0
    assert diagnostics["probability_sum_max_abs_error"] == 0.0
