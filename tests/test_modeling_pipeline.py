from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.dataset import (
    aggregate_article_sentiment,
    build_model_frame,
    load_or_build_model_frame,
)
from src.modeling.hybrid import (
    build_lstm_sequences,
    fit_garch11_baseline,
    validate_garch_fit,
    diebold_mariano_test,
    analyze_forecast_subperiods,
    add_garch_features,
)


def test_aggregate_article_sentiment_builds_daily_features():
    df = pd.DataFrame(
        [
            {
                "trading_date": "2024-01-02",
                "sentiment_score": 0.8,
                "sentiment_label": "positive",
                "url": "url1",
            },
            {
                "trading_date": "2024-01-02",
                "sentiment_score": -0.4,
                "sentiment_label": "negative",
                "url": "url2",
            },
            {
                "trading_date": "2024-01-03",
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "url": "url3",
            },
        ]
    )
    articles_clean = pd.DataFrame(
        [
            {"url": "url1", "category": "Chứng khoán"},
            {"url": "url2", "category": "Vĩ mô"},
            {"url": "url3", "category": "Kinh tế"},
        ]
    )

    daily = aggregate_article_sentiment(df, articles_clean_df=articles_clean)

    assert list(daily.columns) == [
        "date",
        "mean_sentiment",
        "sentiment_std",
        "sentiment_volume",
        "negative_share",
        "neutral_share",
        "positive_share",
        "net_sentiment",
        "sentiment_surprise",
        "macro_sentiment",
        "market_sentiment",
        "macro_sentiment_missing",
        "market_sentiment_missing",
    ]
    assert len(daily) == 2
    assert daily.loc[0, "sentiment_volume"] == 2
    assert daily.loc[0, "positive_share"] == 0.5
    assert daily.loc[0, "negative_share"] == 0.5
    assert daily.loc[0, "net_sentiment"] == 0.0
    assert daily.loc[0, "sentiment_surprise"] == 0.2
    assert daily.loc[1, "sentiment_surprise"] == -0.2
    assert daily.loc[1, "neutral_share"] == 1.0
    # On 2024-01-02, url2 is in "Vĩ mô". So macro_sentiment = -0.4
    assert daily.loc[0, "macro_sentiment"] == -0.4
    # On 2024-01-02, url1 is in "Chứng khoán". So market_sentiment = 0.8
    assert daily.loc[0, "market_sentiment"] == 0.8
    # On 2024-01-03, url3 is in "Kinh tế". So macro_sentiment = 0.0
    assert daily.loc[1, "macro_sentiment"] == 0.0
    # No market category on 2024-01-03, should be NaN (later zero-imputed in frame)
    assert pd.isna(daily.loc[1, "market_sentiment"])
    assert daily.loc[0, "macro_sentiment_missing"] == 0
    assert daily.loc[1, "market_sentiment_missing"] == 1


def test_aggregate_article_sentiment_prefers_embedded_category():
    df = pd.DataFrame(
        [
            {
                "trading_date": "2024-01-02",
                "sentiment_score": 0.4,
                "sentiment_label": "positive",
                "url": "url1",
                "category": "Vĩ mô",
            },
            {
                "trading_date": "2024-01-02",
                "sentiment_score": -0.2,
                "sentiment_label": "negative",
                "url": "url2",
                "category": "Chứng khoán",
            },
        ]
    )
    articles_clean = pd.DataFrame(
        [
            {"url": "url1", "category": "Sai category"},
            {"url": "url2", "category": "Sai category"},
        ]
    )

    daily = aggregate_article_sentiment(df, articles_clean_df=articles_clean)
    assert daily.loc[0, "macro_sentiment"] == 0.4
    assert daily.loc[0, "market_sentiment"] == -0.2


def test_aggregate_article_sentiment_daily_no_drop():
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "mean_sentiment": 0.8,
                "sentiment_std": 0.1,
                "sentiment_volume": 5,
            },
            {
                "date": "2024-01-03",
                "mean_sentiment": -0.4,
                "sentiment_std": 0.2,
                "sentiment_volume": 10,
            },
        ]
    )
    daily = aggregate_article_sentiment(df)
    assert "date" in daily.columns
    assert len(daily) == 2


def test_build_model_frame_merges_sentiment_and_news(tmp_path: Path):
    prices = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "TRDPRC_1": [100.0, 101.0, 100.5],
            "OPEN_PRC": [99.5, 100.0, 100.8],
            "HIGH_1": [101.0, 102.0, 101.0],
            "LOW_1": [99.0, 99.8, 99.7],
            "ACVOL_UNS": [1000, 1200, 1100],
        }
    )
    news = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "n_articles": [5, 6, 4],
            "n_categories": [2, 3, 2],
            "mean_body_len": [500, 550, 520],
        }
    )
    sentiment = pd.DataFrame(
        {
            "trading_date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "sentiment_score": [0.3, -0.1, 0.5],
            "url": ["url1", "url2", "url3"],
        }
    )
    articles_clean = pd.DataFrame(
        {
            "url": ["url1", "url2", "url3"],
            "category": ["Vĩ mô", "Chứng khoán", "Kinh tế"],
        }
    )

    prices_path = tmp_path / "prices.csv"
    news_path = tmp_path / "news.parquet"
    sentiment_path = tmp_path / "sentiment.parquet"
    articles_clean_path = tmp_path / "articles_clean.parquet"

    prices.to_csv(prices_path, index=False)
    news.to_parquet(news_path, index=False)
    sentiment.to_parquet(sentiment_path, index=False)
    articles_clean.to_parquet(articles_clean_path, index=False)

    frame = build_model_frame(
        prices_path,
        daily_news_path=news_path,
        sentiment_path=sentiment_path,
        articles_clean_path=articles_clean_path,
    )

    assert "target_next_vol" in frame.columns
    assert "mean_sentiment" in frame.columns
    assert "macro_sentiment" in frame.columns
    assert "market_sentiment" in frame.columns
    assert "net_sentiment" in frame.columns
    assert "sentiment_surprise" in frame.columns

    assert frame.loc[0, "sentiment_volume"] == 2
    assert frame.loc[0, "macro_sentiment"] == 0.3  # 'url1' category is 'Vĩ mô'
    assert frame.loc[0, "market_sentiment"] == -0.1  # 'url2' category is 'Chứng khoán'

    # 2024-01-04 has no sentiment, should be zero-imputed
    assert frame.loc[2, "has_sentiment"] == 0
    assert frame.loc[2, "macro_sentiment"] == 0.0
    assert frame.loc[2, "market_sentiment"] == 0.0


def test_build_model_frame_accepts_legacy_csv_processed_inputs(tmp_path: Path):
    prices = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "TRDPRC_1": [100.0, 101.0],
            "OPEN_PRC": [99.5, 100.0],
            "HIGH_1": [101.0, 102.0],
            "LOW_1": [99.0, 99.8],
            "ACVOL_UNS": [1000, 1200],
        }
    )
    news = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "n_articles": [2, 0],
            "n_categories": [1, 0],
            "mean_body_len": [400.0, 0.0],
        }
    )
    sentiment = pd.DataFrame(
        [
            {"trading_date": "2024-01-02", "sentiment_score": 0.2, "url": "url1"},
            {"trading_date": "2024-01-02", "sentiment_score": -0.1, "url": "url2"},
        ]
    )
    articles_clean = pd.DataFrame(
        {
            "url": ["url1", "url2"],
            "category": ["Vĩ mô", "Chứng khoán"],
        }
    )

    prices_path = tmp_path / "prices.csv"
    news_path = tmp_path / "daily_news_prices.csv"
    sentiment_path = tmp_path / "article_sentiment_scores.csv"
    articles_clean_path = tmp_path / "articles_clean.csv"

    prices.to_csv(prices_path, index=False)
    news.to_csv(news_path, index=False)
    sentiment.to_csv(sentiment_path, index=False)
    articles_clean.to_csv(articles_clean_path, index=False)

    frame = build_model_frame(
        prices_path,
        daily_news_path=news_path,
        sentiment_path=sentiment_path,
        articles_clean_path=articles_clean_path,
    )

    assert frame.loc[0, "n_articles"] == 2
    assert frame.loc[0, "mean_sentiment"] == 0.05
    assert frame.loc[0, "macro_sentiment"] == 0.2
    assert frame.loc[0, "market_sentiment"] == -0.1


def test_load_or_build_model_frame_prefers_existing_artifact(tmp_path: Path):
    prebuilt = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "log_return": [0.01, -0.02],
            "abs_return": [0.01, 0.02],
            "target_vol": [0.03, 0.04],
            "target_next_vol": [0.04, 0.05],
            "n_articles": [2, 0],
            "n_categories": [1, 0],
            "mean_body_len": [400.0, 0.0],
            "mean_sentiment": [0.1, 0.0],
            "sentiment_std": [0.05, 0.0],
            "sentiment_volume": [2.0, 0.0],
            "negative_share": [0.0, 0.0],
            "neutral_share": [0.5, 1.0],
            "positive_share": [0.5, 0.0],
            "net_sentiment": [0.5, 0.0],
            "sentiment_surprise": [0.1, -0.1],
            "has_sentiment": [1, 0],
            "has_news": [1, 0],
        }
    )
    artifact_path = tmp_path / "modeling_ready.parquet"
    prebuilt.to_parquet(artifact_path, index=False)

    loaded = load_or_build_model_frame(
        model_frame_path=artifact_path,
        price_path=tmp_path / "missing_prices.csv",
    )

    pd.testing.assert_frame_equal(loaded, prebuilt)


def test_fit_garch11_baseline_returns_positive_variance():
    returns = np.array(
        [-0.01, 0.02, -0.015, 0.005, 0.011, -0.006, 0.018, -0.012] * 8,
        dtype=float,
    )

    result = fit_garch11_baseline(returns)

    assert result.omega > 0
    assert result.alpha >= 0
    assert result.beta >= 0
    assert np.all(result.conditional_variance > 0)
    assert len(result.standardized_residuals) == len(returns)


def test_add_garch_features_train_end():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    returns = np.array(
        [-0.01, 0.02, -0.015, 0.005, 0.011, -0.006, 0.018, -0.012] * 13, dtype=float
    )[:100]
    df = pd.DataFrame(
        {
            "date": dates,
            "log_return": returns,
            "target_next_vol": np.abs(returns) * 1.1,
        }
    )

    df_leakfree = add_garch_features(df, train_end="2024-02-15")
    assert "garch_conditional_vol" in df_leakfree.columns
    assert "garch_forecast_vol" in df_leakfree.columns
    assert "garch_std_resid" in df_leakfree.columns
    assert not df_leakfree["garch_forecast_vol"].isna().all()


def test_build_lstm_sequences_splits_temporally():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "feature_a": np.arange(12, dtype=float),
            "feature_b": np.arange(12, dtype=float) / 10,
            "garch_forecast_vol": np.linspace(0.1, 0.2, 12),
            "target_next_vol": np.linspace(0.11, 0.21, 12),
            "hybrid_residual_target": np.linspace(0.01, 0.02, 12),
        }
    )

    sequences, meta = build_lstm_sequences(
        frame,
        feature_columns=["feature_a", "feature_b"],
        target_column="hybrid_residual_target",
        sequence_length=3,
        split_dates=("2024-01-07", "2024-01-09"),
    )

    assert sequences["x_train"].shape[1:] == (3, 2)
    assert meta.train_rows > 0
    assert meta.validation_rows > 0
    assert meta.test_rows > 0


def test_validate_garch_fit():
    # Generate static return series
    returns = np.array(
        [-0.01, 0.02, -0.015, 0.005, 0.011, -0.006, 0.018, -0.012] * 8,
        dtype=float,
    )
    result = fit_garch11_baseline(returns)
    diagnostics = validate_garch_fit(result)

    assert "stationary" in diagnostics
    assert "ljung_box_pvalue_lag5" in diagnostics
    assert "no_remaining_arch_effects" in diagnostics
    assert isinstance(diagnostics["stationary"], bool)


def test_diebold_mariano_test():
    np.random.seed(42)
    actual = np.random.normal(0, 0.01, 50)
    pred1 = actual + np.random.normal(0, 0.005, 50)
    pred2 = actual + np.random.normal(0, 0.002, 50)  # pred2 is better

    dm_stat, p_val = diebold_mariano_test(actual, pred1, pred2, loss_type="square")

    # Since pred2 has smaller errors, pred1_err_sq - pred2_err_sq should be positive, meaning dm_stat > 0
    assert dm_stat > 0
    assert 0.0 <= p_val <= 1.0


def test_analyze_forecast_subperiods():
    actual = np.array([0.01, 0.02, 0.015, 0.03, 0.01, 0.012])
    pred_baseline = np.array([0.011, 0.022, 0.016, 0.033, 0.011, 0.013])
    pred_hybrid = np.array([0.0101, 0.0201, 0.0151, 0.0301, 0.0101, 0.0121])
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    sentiment = np.array([-0.1, 0.1, 0.0, 0.2, -0.2, 0.0])

    analysis = analyze_forecast_subperiods(
        actual, pred_baseline, pred_hybrid, dates, sentiment
    )

    assert "year_2024" in analysis
    assert "shock_regime" in analysis
    assert "calm_regime" in analysis
    assert "negative_sentiment_days" in analysis
    assert "positive_sentiment_days" in analysis

    assert analysis["year_2024"]["size"] == 6
    assert analysis["negative_sentiment_days"]["size"] == 2
    assert analysis["positive_sentiment_days"]["size"] == 2
    assert analysis["year_2024"]["hybrid_rmse"] < analysis["year_2024"]["baseline_rmse"]
