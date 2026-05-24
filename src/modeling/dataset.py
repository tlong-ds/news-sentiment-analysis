"""Dataset builders for the hybrid GARCH plus sentiment-LSTM workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
MACRO_CATEGORIES = {"Vĩ mô", "Kinh tế"}
MARKET_CATEGORIES = {"Chứng khoán", "Thị trường"}


class SentimentAggregationError(ValueError):
    """Raised when a sentiment input file cannot be normalized."""


@dataclass(frozen=True)
class SentimentFrameSpec:
    date_column: str
    is_article_level: bool


def _resolve_category_frame(
    sentiment_df: pd.DataFrame,
    articles_clean_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if "category" in sentiment_df.columns:
        category_frame = sentiment_df[["date", "sentiment_score", "category"]].copy()
        category_frame["category"] = category_frame["category"].fillna("").astype(str)
        return category_frame

    if articles_clean_df is None or "url" not in sentiment_df.columns:
        return None

    merged_df = sentiment_df.merge(articles_clean_df[["url", "category"]], on="url", how="left")
    merged_df["category"] = merged_df["category"].fillna("").astype(str)
    return merged_df[["date", "sentiment_score", "category"]]


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
    sentiment_threshold: float = 0.05,
) -> pd.DataFrame:
    """Aggregate article-level scores to daily trading-day sentiment features."""
    spec = _detect_sentiment_frame(sentiment_df)
    if not spec.is_article_level:
        daily = sentiment_df.copy()
        daily["date"] = pd.to_datetime(daily[spec.date_column])
        if spec.date_column != "date":
            daily = daily.drop(columns=[spec.date_column], errors="ignore")
        return daily.sort_values("date")

    df = sentiment_df.copy()
    df["date"] = pd.to_datetime(df[spec.date_column])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["date", "sentiment_score"])
    if df.empty:
        raise SentimentAggregationError("Sentiment scores are empty after cleaning.")

    label_col = "sentiment_label"
    if label_col not in df.columns or sentiment_threshold != 0.05:
        df["sentiment_label"] = np.select(
            [df["sentiment_score"] > sentiment_threshold, df["sentiment_score"] < -sentiment_threshold],
            ["positive", "negative"],
            default="neutral",
        )

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
    prior_mean = daily["mean_sentiment"].shift(1).rolling(window=5, min_periods=1).mean()
    daily["sentiment_surprise"] = daily["mean_sentiment"] - prior_mean.fillna(0.0)

    # Derived variables: category-specific sentiment (macro & market)
    category_frame = _resolve_category_frame(df, articles_clean_df)
    if category_frame is not None:
        macro_mask = category_frame["category"].isin(MACRO_CATEGORIES)
        macro_daily = (
            category_frame[macro_mask]
            .groupby("date")["sentiment_score"]
            .mean()
            .reset_index(name="macro_sentiment")
        )
        daily = daily.merge(macro_daily, on="date", how="left")

        market_mask = category_frame["category"].isin(MARKET_CATEGORIES)
        market_daily = (
            category_frame[market_mask]
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

    daily["macro_sentiment_missing"] = daily["macro_sentiment"].isna().astype(int)
    daily["market_sentiment_missing"] = daily["market_sentiment"].isna().astype(int)

    return daily.sort_values("date").reset_index(drop=True)


def compute_volatility_features(df: pd.DataFrame, target_type: str = "parkinson") -> pd.DataFrame:
    """Compute returns, Parkinson, Garman-Klass, and z-score features from raw prices."""
    # Ensure columns are standardized
    rename_map = {
        "Date": "date",
        "Close": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
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
    
    if target_type == "parkinson":
        df["target_vol"] = df["parkinson_vol"].fillna(df["abs_return"])
    elif target_type == "garman_klass":
        df["target_vol"] = df["gk_vol"].fillna(df["abs_return"])
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
        
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
    sentiment_threshold: float = 0.05,
    target_type: str = "parkinson",
) -> pd.DataFrame:
    """Merge price, daily news intensity, and daily sentiment into one frame."""
    price_df = pd.read_csv(price_path)
    model_df = compute_volatility_features(price_df, target_type=target_type)

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
            price_parent = Path(price_path).parent
            default_clean_path = price_parent.parent / "processed" / "articles_clean.csv"
            if not default_clean_path.exists():
                default_clean_path = Path("data/processed/articles_clean.csv")
            if default_clean_path.exists():
                articles_clean_df = pd.read_csv(default_clean_path)
                
        daily_sentiment = aggregate_article_sentiment(
            sentiment_df,
            articles_clean_df=articles_clean_df,
            sentiment_threshold=sentiment_threshold,
        )
        model_df = model_df.merge(daily_sentiment, on="date", how="left")

    # Step 4.2: Zero-imputation handling for trading days with zero articles.
    # Count of zero-news trading days is 4 out of 2498 (0.16%). Since this is
    # well under the 5% threshold, zero-imputation is a highly defensible
    # approximation for missing sentiment control features.
    zero_news_mask = model_df["sentiment_volume"].fillna(0).eq(0) if "sentiment_volume" in model_df.columns else pd.Series(False, index=model_df.index)
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
    }
    for column, default in fill_defaults.items():
        if column in model_df.columns:
            model_df[column] = model_df[column].fillna(default)

    for column in ["macro_sentiment", "market_sentiment"]:
        if column in model_df.columns:
            model_df.loc[zero_news_mask, column] = model_df.loc[zero_news_mask, column].fillna(0.0)

    for column in ["macro_sentiment_missing", "market_sentiment_missing"]:
        if column in model_df.columns:
            model_df[column] = model_df[column].fillna(1).astype(int)

    model_df["has_sentiment"] = (
        model_df["sentiment_volume"].gt(0).astype(int)
        if "sentiment_volume" in model_df.columns
        else 0
    )
    model_df["has_news"] = model_df["has_sentiment"]
    return model_df.sort_values("date").reset_index(drop=True)
