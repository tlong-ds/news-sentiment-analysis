"""Generate additional figures for the LaTeX report results section (sentiment bias and robustness comparison)."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Style settings
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


def plot_sentiment_distribution(sentiment_path: str | Path, output_dir: Path) -> None:
    """Plot the distribution of individual article sentiment scores to show positive bias."""
    df = pd.read_parquet(sentiment_path)
    scores = df["sentiment_score"].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Histogram of scores
    counts, bins, patches = ax.hist(
        scores,
        bins=50,
        alpha=0.7,
        color="#2ca02c",
        edgecolor="#1e6f1e",
        label="Sentiment Scores",
    )

    # Add vertical lines for thresholds
    ax.axvline(
        0.05,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="Positive Threshold (0.05)",
    )
    ax.axvline(
        -0.05,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label="Negative Threshold (-0.05)",
    )

    # Count proportions
    pos_pct = (scores > 0.05).mean() * 100
    neg_pct = (scores < -0.05).mean() * 100
    neu_pct = ((scores >= -0.05) & (scores <= 0.05)).mean() * 100

    ax.set_title("CafeF News Sitemap Sentiment Score Distribution")
    ax.set_xlabel("PhoBERT Sentiment Score")
    ax.set_ylabel("Number of Articles")

    # Annotation box for shares
    text_str = (
        f"Positive (>0.05): {pos_pct:.1f}%\n"
        f"Neutral ([-0.05, 0.05]): {neu_pct:.1f}%\n"
        f"Negative (<-0.05): {neg_pct:.1f}%\n\n"
        f"Total Articles: {len(scores):,}"
    )
    ax.text(
        0.05,
        0.65,
        text_str,
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5"),
    )

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "sentiment_distribution.png", dpi=300)
    plt.close(fig)
    logger.info("Saved sentiment_distribution.png")


def plot_robustness_comparison(summary_path: str | Path, output_dir: Path) -> None:
    """Plot GARCH vs Hybrid RMSE across all robustness specifications."""
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    specs = {
        "Baseline": "baseline",
        "Threshold 0.10": "threshold_0.10",
        "Threshold 0.15": "threshold_0.15",
        "Garman-Klass": "garman_klass",
        "Ablation": "no_sentiment_ablation",
        "GARCH-X": "garch_x",
        "Expanding GARCH": "expanding_garch",
    }

    labels = []
    garch_rmse = []
    hybrid_rmse = []

    for label, key in specs.items():
        if key in summary:
            labels.append(label)
            # Re-scale to percent volatility if actual values are small, or keep absolute
            garch_rmse.append(
                summary[key]["baseline_rmse"] * 100
            )  # scale to percentage points
            hybrid_rmse.append(summary[key]["hybrid_rmse"] * 100)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(
        x - width / 2,
        garch_rmse,
        width,
        label="GARCH Baseline",
        color="#d62728",
        alpha=0.85,
    )
    rects2 = ax.bar(
        x + width / 2,
        hybrid_rmse,
        width,
        label="Hybrid Model",
        color="#1f77b4",
        alpha=0.85,
    )

    ax.set_ylabel("Out-of-Sample RMSE (%)")
    ax.set_title(
        "Volatility Forecast Performance (RMSE) Across Robustness Specifications"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    fig.savefig(output_dir / "robustness_comparison.png", dpi=300)
    plt.close(fig)
    logger.info("Saved robustness_comparison.png")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    sentiment_path = "data/sentiment/article_sentiment_scores.parquet"
    summary_path = "data/interim/robustness_experiment_summary.json"

    output_dir = Path("report/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_sentiment_distribution(sentiment_path, output_dir)
    plot_robustness_comparison(summary_path, output_dir)


if __name__ == "__main__":
    main()
