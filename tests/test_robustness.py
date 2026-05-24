from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.dataset import aggregate_article_sentiment, compute_volatility_features
from src.modeling.hybrid import fit_garchx11_baseline, fit_expanding_garch


def test_custom_sentiment_threshold_recomputation():
    sentiment_df = pd.DataFrame(
        [
            {"trading_date": "2024-01-02", "sentiment_score": 0.08, "url": "url1"},
            {"trading_date": "2024-01-02", "sentiment_score": -0.04, "url": "url2"},
            {"trading_date": "2024-01-03", "sentiment_score": 0.12, "url": "url3"},
        ]
    )
    # Threshold = 0.05 (Default) -> score 0.08 positive, -0.04 neutral, 0.12 positive
    daily_05 = aggregate_article_sentiment(sentiment_df, sentiment_threshold=0.05)
    assert daily_05.loc[0, "positive_share"] == 0.5
    assert daily_05.loc[0, "neutral_share"] == 0.5
    assert daily_05.loc[0, "negative_share"] == 0.0

    # Threshold = 0.10 -> score 0.08 neutral, -0.04 neutral, 0.12 positive
    daily_10 = aggregate_article_sentiment(sentiment_df, sentiment_threshold=0.10)
    assert daily_10.loc[0, "positive_share"] == 0.0
    assert daily_10.loc[0, "neutral_share"] == 1.0
    assert daily_10.loc[0, "negative_share"] == 0.0
    assert daily_10.loc[1, "positive_share"] == 1.0


def test_garman_klass_target_volatility():
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
    # Parkinson
    df_parkinson = compute_volatility_features(prices, target_type="parkinson")
    assert "gk_vol" in df_parkinson.columns
    assert "parkinson_vol" in df_parkinson.columns
    
    # Garman-Klass
    df_gk = compute_volatility_features(prices, target_type="garman_klass")
    # Verify target_vol is filled with gk_vol (not Parkinson)
    assert np.allclose(df_gk["target_vol"].dropna(), df_gk["gk_vol"].dropna())


def test_garch_x_fitting():
    np.random.seed(42)
    n_days = 100
    # Simulate stationary returns and exogenous sentiment variable
    returns = np.random.normal(0, 0.01, n_days)
    exog = np.random.uniform(-1, 1, n_days)
    
    result = fit_garchx11_baseline(returns, exog, scale=100.0)
    
    assert result.omega > 0
    assert result.alpha >= 0
    assert result.beta >= 0
    assert result.alpha + result.beta < 1.0
    assert len(result.conditional_variance) == n_days
    assert len(result.variance_forecast) == n_days
    assert len(result.standardized_residuals) == n_days
    assert hasattr(result, "gamma")


def test_fit_expanding_garch():
    np.random.seed(42)
    n_days = 80
    returns = np.random.normal(0, 0.01, n_days)
    
    train_len = 50
    cond_vol, forecast_vol, std_resid = fit_expanding_garch(
        returns,
        train_len=train_len,
        reestimate_freq=10,
        scale=100.0,
    )
    
    assert len(cond_vol) == n_days
    assert len(forecast_vol) == n_days
    assert len(std_resid) == n_days
    assert not np.any(np.isnan(cond_vol))
    assert not np.any(np.isnan(forecast_vol))
    assert not np.any(np.isnan(std_resid))
