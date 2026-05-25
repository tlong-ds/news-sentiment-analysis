"""Generate figures for the LaTeX report."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.modeling.dataset import compute_volatility_features
from src.modeling.hybrid import add_garch_features

logger = logging.getLogger(__name__)

# Use standard clean plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "legend.fontsize": 10,
    }
)


def plot_return_distribution(prices_path: str | Path, output_dir: Path) -> None:
    """Plot return histogram with a normal curve overlay to show fat tails."""
    df = pd.read_csv(prices_path)
    # Standardize headers
    df = df.rename(columns={"Date": "date", "Close": "close", "TRDPRC_1": "close"})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    returns = df["log_return"].dropna().to_numpy() * 100.0  # in percent

    fig, ax = plt.subplots(figsize=(7, 4.5))
    count, bins, ignored = ax.hist(
        returns,
        bins=60,
        density=True,
        alpha=0.6,
        color="#1f77b4",
        edgecolor="#15537e",
        label="Returns",
    )

    # Fit normal distribution
    mu, std = norm.fit(returns)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, "-", linewidth=2, color="#d62728", label="Normal Fit")

    # Calculate kurtosis
    kurtosis = pd.Series(returns).kurtosis()

    ax.set_title("VN-Index Daily Return Distribution vs. Normal Fit")
    ax.set_xlabel("Daily Log Return (%)")
    ax.set_ylabel("Probability Density")
    ax.text(
        0.05,
        0.85,
        f"Mean: {mu:.4f}%\nStd Dev: {std:.4f}%\nKurtosis: {kurtosis:.2f} (Fat Tails)",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "return_distribution.png", dpi=300)
    plt.close(fig)
    logger.info("Saved return_distribution.png")


def plot_volatility_clustering(prices_path: str | Path, output_dir: Path) -> None:
    """Plot returns alongside GARCH conditional volatility to demonstrate clustering."""
    df = pd.read_csv(prices_path)
    df = compute_volatility_features(df)

    # Run GARCH on full sample to show the clustering
    model_df = add_garch_features(df)
    dates = pd.to_datetime(model_df["date"])
    returns = model_df["log_return"].to_numpy() * 100.0  # %
    vol = model_df["garch_conditional_vol"].to_numpy() * 100.0  # %

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color = "#888888"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily Return (%)", color=color)
    ax1.plot(dates, returns, color=color, alpha=0.5, linewidth=0.5, label="Log Return")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#1f77b4"
    ax2.set_ylabel("GARCH Conditional Volatility (%)", color=color)
    ax2.plot(dates, vol, color=color, linewidth=1.2, label="GARCH(1,1) Volatility")
    ax2.tick_params(axis="y", labelcolor=color)

    # Highlight historical periods
    # COVID crash (early 2020)
    ax2.axvspan(
        pd.Timestamp("2020-03-01"),
        pd.Timestamp("2020-05-31"),
        color="#d62728",
        alpha=0.1,
    )
    ax2.text(
        pd.Timestamp("2020-04-15"),
        vol.max() * 0.9,
        "COVID-19",
        color="#d62728",
        weight="bold",
        ha="center",
    )

    # Liquidity crash (April 2022)
    ax2.axvspan(
        pd.Timestamp("2022-04-01"),
        pd.Timestamp("2022-06-30"),
        color="#d62728",
        alpha=0.1,
    )
    ax2.text(
        pd.Timestamp("2022-05-15"),
        vol.max() * 0.9,
        "Liquidity Shock",
        color="#d62728",
        weight="bold",
        ha="center",
    )

    plt.title("Volatility Clustering on VN-Index (2015-2024)")
    fig.tight_layout()
    fig.savefig(output_dir / "vol_clustering.png", dpi=300)
    plt.close(fig)
    logger.info("Saved vol_clustering.png")


def plot_forecast_comparison(summary_path: str | Path, output_dir: Path) -> None:
    """Plot realized volatility vs. GARCH baseline and GARCH + LSTM forecasts on the test set."""
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    dates = pd.to_datetime(summary["test_dates"])
    actual = np.asarray(summary["actual_volatility"]) * 100.0  # %
    baseline = np.asarray(summary["baseline_forecast"]) * 100.0  # %
    hybrid = np.asarray(summary["hybrid_forecast"]) * 100.0  # %

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        dates,
        actual,
        color="#7f7f7f",
        alpha=0.7,
        linewidth=1.0,
        label="Realized Parkinson Volatility",
    )
    ax.plot(
        dates,
        baseline,
        color="#d62728",
        alpha=0.9,
        linewidth=1.2,
        linestyle="--",
        label="GARCH(1,1) Baseline",
    )
    ax.plot(
        dates,
        hybrid,
        color="#1f77b4",
        alpha=0.9,
        linewidth=1.5,
        label="Hybrid GARCH + Sentiment LSTM",
    )

    ax.set_title("Out-of-Sample Volatility Forecast Comparison (2023-2024)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "forecast_comparison.png", dpi=300)
    plt.close(fig)
    logger.info("Saved forecast_comparison.png")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    prices = "data/main/raw/prices_VN.csv"
    summary = "data/main/processed/hybrid_experiment_summary.json"

    output_dir = Path("report/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_return_distribution(prices, output_dir)
    plot_volatility_clustering(prices, output_dir)
    plot_forecast_comparison(summary, output_dir)


if __name__ == "__main__":
    main()
