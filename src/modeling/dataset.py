"""Dataset builders for the hybrid GARCH plus sentiment-LSTM workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentAggregationError(ValueError):
    """Raised when a sentiment input file cannot be normalized."""


@dataclass(frozen=True)
class SentimentFrameSpec:
    date_column: str
    is_article_level: bool


def _detect_sentiment_frame(df: pd.DataFrame) -> SentimentFrameSpec:
    columns = set(df.columns)
    date_candidates = ["trading_date", "date"]
    date_column = next((col for col in date_candidates if col in columns), None)
    if date_column is None:
        raise SentimentAggregationError(
            "Sentiment data must contain either 'trading_date' or 'date'."
        )

    has_score = "sentiment_score" in columns
    daily_cols = {"mean_sentiment", "sentiment_std", "sentiment_volume"}

    if has_score:
        return SentimentFrameSpec(date_column=date_column, is_article_level=True)
    if daily_cols & columns:
        return SentimentFrameSpec(date_column=date_column, is_article_level=False)

    raise SentimentAggregationError(
        "Sentiment data must contain article-level 'sentiment_score' or daily "
        "aggregates such as 'mean_sentiment'."
    )


def aggregate_article_sentiment(
    sentiment_df: pd.DataFrame,
    articles_clean_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate article-level scores to daily trading-day sentiment features."""
    spec = _detect_sentiment_frame(sentiment_df)
    if not spec.is_article_level:
        daily = sentiment_df.copy()
        daily["date"] = pd.to_datetime(daily[spec.date_column])
        return daily.drop(columns=[spec.date_column], errors="ignore").sort_values("date")

    df = sentiment_df.copy()
    df["date"] = pd.to_datetime(df[spec.date_column])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["date", "sentiment_score"])
    if df.empty:
        raise SentimentAggregationError("Sentiment scores are empty after cleaning.")

    label_col = "sentiment_label" if "sentiment_label" in df.columns else None
    if label_col is None:
        df["sentiment_label"] = np.select(
            [df["sentiment_score"] > 0.05, df["sentiment_score"] < -0.05],
            ["positive", "negative"],
            default="neutral",
        )
        label_col = "sentiment_label"

    grouped = df.groupby("date", as_index=False).agg(
        mean_sentiment=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        sentiment_volume=("sentiment_score", "size"),
    )

    label_shares = (
        df.assign(value=1)
        .pivot_table(
            index="date",
            columns=label_col,
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
        .rename_axis(columns=None)
        .reset_index()
    )

    for label in ["negative", "neutral", "positive"]:
        if label not in label_shares.columns:
            label_shares[label] = 0

    label_shares["total_labels"] = label_shares[["negative", "neutral", "positive"]].sum(axis=1)
    for label in ["negative", "neutral", "positive"]:
        label_shares[f"{label}_share"] = label_shares[label] / label_shares["total_labels"]

    daily = grouped.merge(
        label_shares[
            [
                "date",
                "negative_share",
                "neutral_share",
                "positive_share",
            ]
        ],
        on="date",
        how="left",
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)

    # Derived variable: net_sentiment = positive_share - negative_share
    daily["net_sentiment"] = daily["positive_share"] - daily["negative_share"]

    # Derived variable: sentiment_surprise = mean_sentiment - rolling_5day_mean_sentiment
    daily["sentiment_surprise"] = (
        daily["mean_sentiment"] -
        daily["mean_sentiment"].rolling(window=5, min_periods=1).mean()
    )

    # Derived variables: category-specific sentiment (macro & market)
    if articles_clean_df is not None:
        # Merge article categories into the scores frame using 'url'
        merged_df = df.merge(articles_clean_df[["url", "category"]], on="url", how="left")
        
        # Macro sentiment: mean sentiment of categories 'Vĩ mô' and 'Kinh tế'
        macro_mask = merged_df["category"].isin(["Vĩ mô", "Kinh tế"])
        macro_daily = (
            merged_df[macro_mask]
            .groupby("date")["sentiment_score"]
            .mean()
            .reset_index(name="macro_sentiment")
        )
        daily = daily.merge(macro_daily, on="date", how="left")
        
        # Market sentiment: mean sentiment of categories 'Chứng khoán' and 'Thị trường'
        market_mask = merged_df["category"].isin(["Chứng khoán", "Thị trường"])
        market_daily = (
            merged_df[market_mask]
            .groupby("date")["sentiment_score"]
            .mean()
            .reset_index(name="market_sentiment")
        )
        daily = daily.merge(market_daily, on="date", how="left")
    else:
        logger.warning(
            "articles_clean_df not provided. 'macro_sentiment' and 'market_sentiment' "
            "cannot be computed and will default to NaN (then zero-imputed)."
        )
        daily["macro_sentiment"] = np.nan
        daily["market_sentiment"] = np.nan

    return daily.sort_values("date").reset_index(drop=True)


def compute_volatility_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Create return and volatility proxy features from VN-Index OHLCV data."""
    df = price_df.copy()
    rename_map = {
        "Date": "date",
        "TRDPRC_1": "close",
        "OPEN_PRC": "open",
        "HIGH_1": "high",
        "LOW_1": "low",
        "ACVOL_UNS": "volume",
    }
    df = df.rename(columns=rename_map)
    required = {"date", "close", "open", "high", "low", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price data missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["abs_return"] = df["log_return"].abs()
    df["squared_return"] = df["log_return"].pow(2)
    log_high_low = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
    df["parkinson_vol"] = np.sqrt(log_high_low.pow(2) / (4 * np.log(2)))
    df["gk_vol"] = np.sqrt(
        np.maximum(
            0.0,
            0.5 * log_high_low.pow(2)
            - (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]).pow(2),
        )
    )
    df["target_vol"] = df["parkinson_vol"].fillna(df["abs_return"])
    df["target_next_vol"] = df["target_vol"].shift(-1)
    df["volume_zscore_21"] = (
        (df["volume"] - df["volume"].rolling(21).mean()) / df["volume"].rolling(21).std()
    )
    return df


def build_model_frame(
    price_path: str | Path,
    *,
    daily_news_path: str | Path | None = None,
    sentiment_path: str | Path | None = None,
    articles_clean_path: str | Path | None = None,
) -> pd.DataFrame:
    """Merge price, daily news intensity, and daily sentiment into one frame."""
    price_df = pd.read_csv(price_path)
    model_df = compute_volatility_features(price_df)

    if daily_news_path is not None:
        daily_news = pd.read_csv(daily_news_path, parse_dates=["date"])
        keep_cols = [
            col
            for col in ["date", "n_articles", "n_categories", "mean_body_len"]
            if col in daily_news.columns
        ]
        model_df = model_df.merge(daily_news[keep_cols], on="date", how="left")

    if sentiment_path is not None:
        sentiment_df = pd.read_csv(sentiment_path)
        
        # Load articles_clean.csv to extract categories for macro/market sentiment
        articles_clean_df = None
        if articles_clean_path is not None:
            articles_clean_df = pd.read_csv(articles_clean_path)
        else:
            # Fall back to default location relative to price_path
            default_clean_path = Path(price_path).parent / "processed" / "articles_clean.csv"
            if default_clean_path.exists():
                articles_clean_df = pd.read_csv(default_clean_path)
                
        daily_sentiment = aggregate_article_sentiment(sentiment_df, articles_clean_df=articles_clean_df)
        model_df = model_df.merge(daily_sentiment, on="date", how="left")

    # Step 4.2: Zero-imputation handling for trading days with zero articles.
    # Count of zero-news trading days is 4 out of 2498 (0.16%). Since this is
    # well under the 5% threshold, zero-imputation is a highly defensible
    # approximation for missing sentiment control features.
    fill_defaults = {
        "n_articles": 0,
        "n_categories": 0,
        "mean_body_len": 0.0,
        "mean_sentiment": 0.0,
        "sentiment_std": 0.0,
        "sentiment_volume": 0.0,
        "negative_share": 0.0,
        "neutral_share": 0.0,
        "positive_share": 0.0,
        "net_sentiment": 0.0,
        "sentiment_surprise": 0.0,
        "macro_sentiment": 0.0,
        "market_sentiment": 0.0,
    }
    for column, default in fill_defaults.items():
        if column in model_df.columns:
            model_df[column] = model_df[column].fillna(default)

    model_df["has_sentiment"] = (
        model_df["sentiment_volume"].gt(0).astype(int)
        if "sentiment_volume" in model_df.columns
        else 0
    )
    return model_df.sort_values("date").reset_index(drop=True)
