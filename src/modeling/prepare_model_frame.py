"""Build a modeling-ready parquet from prices, news, and sentiment artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.modeling.dataset import build_model_frame


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the merged modeling frame used by volatility scripts."
    )
    parser.add_argument("--prices", default="data/main/raw/prices_VN.csv")
    parser.add_argument(
        "--daily-news", default="data/main/processed/daily_news_prices.parquet"
    )
    parser.add_argument(
        "--sentiment", default="data/main/processed/article_sentiment_scores.parquet"
    )
    parser.add_argument(
        "--articles-clean", default="data/main/processed/articles_clean.parquet"
    )
    parser.add_argument(
        "--output-file", default="data/main/processed/modeling_ready.parquet"
    )
    parser.add_argument("--target-type", default="parkinson")
    parser.add_argument("--sentiment-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    model_df = build_model_frame(
        args.prices,
        daily_news_path=args.daily_news,
        sentiment_path=args.sentiment,
        articles_clean_path=args.articles_clean,
        sentiment_threshold=args.sentiment_threshold,
        target_type=args.target_type,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_parquet(output_path, index=False)

    report = {
        "rows": int(len(model_df)),
        "date_min": str(model_df["date"].min()),
        "date_max": str(model_df["date"].max()),
        "columns": model_df.columns.tolist(),
        "output_file": str(output_path),
    }
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved modeling-ready frame -> %s", output_path)


if __name__ == "__main__":
    main()
