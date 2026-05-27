"""Compute descriptive statistics and GARCH diagnostics for VN-Index returns and news sentiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, chi2

from src.utils.io import read_parquet_table

logger = logging.getLogger(__name__)


def ljung_box_test(x: np.ndarray, lags: list[int]) -> dict[int, tuple[float, float]]:
    """Manual Ljung-Box Q-test for autocorrelation."""
    n = len(x)
    mean_x = np.mean(x)
    denom = np.sum((x - mean_x) ** 2)

    if denom == 0:
        return {lag: (0.0, 1.0) for lag in lags}

    autocorr = []
    for k in range(1, max(lags) + 1):
        num = np.sum((x[k:] - mean_x) * (x[:-k] - mean_x))
        autocorr.append(num / denom)

    results = {}
    for lag in lags:
        q_stat = 0.0
        for k in range(1, lag + 1):
            q_stat += (autocorr[k - 1] ** 2) / (n - k)
        q_stat *= n * (n + 2)
        p_val = 1.0 - chi2.cdf(q_stat, df=lag)
        results[lag] = (q_stat, p_val)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute descriptive statistics and write a LaTeX table."
    )
    parser.add_argument("--prices", default="data/raw/prices_VN.csv")
    parser.add_argument(
        "--articles-clean", default="data/interim/articles_clean.parquet"
    )
    parser.add_argument(
        "--sentiment",
        default="data/sentiment/article_sentiment_scores.parquet",
    )
    parser.add_argument(
        "--daily-news", default="data/interim/daily_news_prices.parquet"
    )
    parser.add_argument("--output-dir", default="report/tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- Loading Datasets ---")
    df_prices = pd.read_csv(args.prices)
    df_prices = df_prices.rename(
        columns={"Date": "date", "Close": "close", "TRDPRC_1": "close"}
    )
    df_prices["log_return"] = np.log(df_prices["close"] / df_prices["close"].shift(1))
    returns = df_prices["log_return"].dropna().to_numpy() * 100.0

    df_sent = read_parquet_table(args.sentiment)
    df_daily = read_parquet_table(args.daily_news)
    df_art = read_parquet_table(args.articles_clean)

    logger.info("--- VN-Index Returns Descriptive Statistics ---")
    n_days = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    min_ret = np.min(returns)
    max_ret = np.max(returns)

    ret_series = pd.Series(returns)
    skew_ret = ret_series.skew()
    kurt_ret = ret_series.kurtosis() + 3.0
    excess_kurt_ret = ret_series.kurtosis()

    jb_stat, jb_pval = jarque_bera(returns)

    lb_ret = ljung_box_test(returns, lags=[5, 10])
    squared_returns = returns**2
    lb_sq_ret = ljung_box_test(squared_returns, lags=[5, 10])

    logger.info("Trading Days: %d", n_days)
    logger.info("Mean Return: %.6f%%", mean_ret)
    logger.info("Std Dev Return: %.6f%%", std_ret)
    logger.info("Min Return: %.6f%%", min_ret)
    logger.info("Max Return: %.6f%%", max_ret)
    logger.info("Skewness: %.6f", skew_ret)
    logger.info("Kurtosis (Total): %.6f (Excess: %.6f)", kurt_ret, excess_kurt_ret)
    logger.info("Jarque-Bera Stat: %.4f (p-value: %.6e)", jb_stat, jb_pval)
    logger.info(
        "Ljung-Box (Return, Lag 5): Q=%.4f, p=%.6f",
        lb_ret[5][0],
        lb_ret[5][1],
    )
    logger.info(
        "Ljung-Box (Return, Lag 10): Q=%.4f, p=%.6f",
        lb_ret[10][0],
        lb_ret[10][1],
    )
    logger.info(
        "Ljung-Box (Sq Return, Lag 5): Q=%.4f, p=%.6e",
        lb_sq_ret[5][0],
        lb_sq_ret[5][1],
    )
    logger.info(
        "Ljung-Box (Sq Return, Lag 10): Q=%.4f, p=%.6e",
        lb_sq_ret[10][0],
        lb_sq_ret[10][1],
    )

    logger.info("--- News and Sentiment Statistics ---")
    total_articles = len(df_art)
    n_sent_articles = len(df_sent)
    mean_sent = df_sent["sentiment_score"].mean()
    std_sent = df_sent["sentiment_score"].fillna(0.0).std()
    min_sent = df_sent["sentiment_score"].min()
    max_sent = df_sent["sentiment_score"].max()

    pos_articles = len(df_sent[df_sent["sentiment_label"] == "positive"])
    neg_articles = len(df_sent[df_sent["sentiment_label"] == "negative"])
    neu_articles = len(df_sent[df_sent["sentiment_label"] == "neutral"])

    logger.info("Total Articles Cleaned: %d", total_articles)
    logger.info("Articles with Sentiment: %d", n_sent_articles)
    logger.info("Mean Sentiment Score: %.6f", mean_sent)
    logger.info("Std Dev Sentiment Score: %.6f", std_sent)
    logger.info("Min Sentiment Score: %.6f", min_sent)
    logger.info("Max Sentiment Score: %.6f", max_sent)
    logger.info(
        "Positive Articles: %d (%.2f%%)",
        pos_articles,
        pos_articles / n_sent_articles * 100,
    )
    logger.info(
        "Negative Articles: %d (%.2f%%)",
        neg_articles,
        neg_articles / n_sent_articles * 100,
    )
    logger.info(
        "Neutral Articles: %d (%.2f%%)",
        neu_articles,
        neu_articles / n_sent_articles * 100,
    )

    daily_articles = df_daily["n_articles"]
    logger.info(
        "Daily Article Volume: Mean=%.2f, Std=%.2f, Min=%.0f, Max=%.0f",
        daily_articles.mean(),
        daily_articles.std(),
        daily_articles.min(),
        daily_articles.max(),
    )
    zero_news_days = (daily_articles == 0).sum()
    logger.info(
        "Zero-news trading days: %d (%.4f%%)",
        zero_news_days,
        (zero_news_days / len(df_daily)) * 100,
    )

    logger.info("Category Distribution:")
    cat_counts = df_art["category"].value_counts()
    for cat, count in cat_counts.items():
        logger.info("  - %s: %d (%.2f%%)", cat, count, count / total_articles * 100)

    latex_table = f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Descriptive Statistics of VN-Index Daily Returns and News Sentiment}}
    \\label{{tab:descriptive_stats}}
    \\begin{{tabular}}{{@{{}}lccccc@{{}}}}
    \\toprule
    Series & Mean & Std. Dev. & Minimum & Maximum & Kurtosis \\\\
    \\midrule
    VN-Index Returns (\\%) & {mean_ret:.4f} & {std_ret:.4f} & {min_ret:.4f} & {max_ret:.4f} & {kurt_ret:.4f} \\\\
    News Sentiment Score & {mean_sent:.4f} & {std_sent:.4f} & {min_sent:.4f} & {max_sent:.4f} & -- \\\\
    Daily News Volume & {daily_articles.mean():.1f} & {daily_articles.std():.1f} & {daily_articles.min():.0f} & {daily_articles.max():.0f} & -- \\\\
    \\bottomrule
    \\end{{tabular}}
\\end{{table}}
"""

    out_path = output_dir / "descriptive_stats_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    logger.info("LaTeX table written -> %s", out_path)


if __name__ == "__main__":
    main()
