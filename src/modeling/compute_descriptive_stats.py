"""Compute descriptive statistics and GARCH diagnostics for VN-Index returns and news sentiment."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, chi2


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


def main() -> None:
    # 1. Paths
    prices_path = Path("data/main/raw/prices_VN.csv")
    articles_path = Path("data/main/processed/articles_clean.parquet")
    sentiment_path = Path("data/main/processed/article_sentiment_scores.parquet")
    daily_news_path = Path("data/main/processed/daily_news_prices.parquet")

    output_dir = Path("report/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Loading Datasets ---")
    df_prices = pd.read_csv(prices_path)
    # Rename columns to standard
    df_prices = df_prices.rename(
        columns={"Date": "date", "Close": "close", "TRDPRC_1": "close"}
    )

    # Calculate returns in percent
    df_prices["log_return"] = np.log(df_prices["close"] / df_prices["close"].shift(1))
    returns = df_prices["log_return"].dropna().to_numpy() * 100.0

    # Load sentiment scores
    df_sent = pd.read_parquet(sentiment_path)

    # Load daily aggregates
    df_daily = pd.read_parquet(daily_news_path)

    # Load clean articles
    df_art = pd.read_parquet(articles_path)

    print("\n--- VN-Index Returns Descriptive Statistics ---")
    n_days = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    min_ret = np.min(returns)
    max_ret = np.max(returns)

    # Skewness and Kurtosis
    ret_series = pd.Series(returns)
    skew_ret = ret_series.skew()
    kurt_ret = ret_series.kurtosis() + 3.0  # Excess to total kurtosis
    excess_kurt_ret = ret_series.kurtosis()

    # Jarque-Bera
    jb_stat, jb_pval = jarque_bera(returns)

    # Ljung-Box on Returns
    lb_ret = ljung_box_test(returns, lags=[5, 10])

    # Ljung-Box on Squared Returns (ARCH effects)
    squared_returns = returns**2
    lb_sq_ret = ljung_box_test(squared_returns, lags=[5, 10])

    print(f"Trading Days: {n_days}")
    print(f"Mean Return: {mean_ret:.6f}%")
    print(f"Std Dev Return: {std_ret:.6f}%")
    print(f"Min Return: {min_ret:.6f}%")
    print(f"Max Return: {max_ret:.6f}%")
    print(f"Skewness: {skew_ret:.6f}")
    print(f"Kurtosis (Total): {kurt_ret:.6f} (Excess: {excess_kurt_ret:.6f})")
    print(f"Jarque-Bera Stat: {jb_stat:.4f} (p-value: {jb_pval:.6e})")
    print(f"Ljung-Box (Return, Lag 5): Q={lb_ret[5][0]:.4f}, p={lb_ret[5][1]:.6f}")
    print(f"Ljung-Box (Return, Lag 10): Q={lb_ret[10][0]:.4f}, p={lb_ret[10][1]:.6f}")
    print(
        f"Ljung-Box (Sq Return, Lag 5): Q={lb_sq_ret[5][0]:.4f}, p={lb_sq_ret[5][1]:.6e}"
    )
    print(
        f"Ljung-Box (Sq Return, Lag 10): Q={lb_sq_ret[10][0]:.4f}, p={lb_sq_ret[10][1]:.6e}"
    )

    print("\n--- News Sitemap and Sentiment Statistics ---")
    total_articles = len(df_art)
    n_sent_articles = len(df_sent)
    mean_sent = df_sent["sentiment_score"].mean()
    std_sent = df_sent["sentiment_score"].fillna(0.0).std()
    min_sent = df_sent["sentiment_score"].min()
    max_sent = df_sent["sentiment_score"].max()

    pos_articles = len(df_sent[df_sent["sentiment_label"] == "positive"])
    neg_articles = len(df_sent[df_sent["sentiment_label"] == "negative"])
    neu_articles = len(df_sent[df_sent["sentiment_label"] == "neutral"])

    print(f"Total Articles Cleaned: {total_articles}")
    print(f"Articles with Sentiment: {n_sent_articles}")
    print(f"Mean Sentiment Score: {mean_sent:.6f}")
    print(f"Std Dev Sentiment Score: {std_sent:.6f}")
    print(f"Min Sentiment Score: {min_sent:.6f}")
    print(f"Max Sentiment Score: {max_sent:.6f}")
    print(
        f"Positive Articles: {pos_articles} ({pos_articles / n_sent_articles * 100:.2f}%)"
    )
    print(
        f"Negative Articles: {neg_articles} ({neg_articles / n_sent_articles * 100:.2f}%)"
    )
    print(
        f"Neutral Articles: {neu_articles} ({neu_articles / n_sent_articles * 100:.2f}%)"
    )

    # Daily volume of articles
    daily_articles = df_daily["n_articles"]
    print(
        f"\nDaily Article Volume: Mean={daily_articles.mean():.2f}, Std={daily_articles.std():.2f}, Min={daily_articles.min():.0f}, Max={daily_articles.max():.0f}"
    )

    # Zero-news days
    zero_news_days = (daily_articles == 0).sum()
    print(
        f"Zero-news trading days: {zero_news_days} ({(zero_news_days / len(df_daily)) * 100:.4f}%)"
    )

    # Category analysis
    print("\nCategory Distribution:")
    cat_counts = df_art["category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"  - {cat}: {count} ({count / total_articles * 100:.2f}%)")

    # Write LaTeX descriptive statistics table
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

    with open(output_dir / "descriptive_stats_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\nLaTeX table written to {output_dir / 'descriptive_stats_table.tex'}")


if __name__ == "__main__":
    main()
