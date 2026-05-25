"""Run classifier inference on CafeF article inputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, TFRobertaForSequenceClassification

from src.config import CAFEF_DATA_DIR, MODELS_DATA_DIR, PROCESSED_DATA_DIR
from src.sentiment.common import ensure_parent_dir
from src.utils.io import read_parquet_table

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PhoBERT classifier inference on CafeF inputs.")
    parser.add_argument("--model-dir", default=f"{MODELS_DATA_DIR}/phobert-sentiment-cafef")
    parser.add_argument("--input-file", default=f"{CAFEF_DATA_DIR}/cafef_input.parquet")
    parser.add_argument("--output-file", default=f"{PROCESSED_DATA_DIR}/article_sentiment_scores.parquet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    return parser.parse_args()


def predict_probabilities(
    texts: list[str],
    *,
    model: TFRobertaForSequenceClassification,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> np.ndarray:
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="tf",
    )
    logits = model(encodings).logits.numpy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def build_output_rows(batch_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    prob_negative = probabilities[:, 0]
    prob_neutral = probabilities[:, 1]
    prob_positive = probabilities[:, 2]
    labels = np.argmax(probabilities, axis=1)
    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return pd.DataFrame(
        {
            "url": batch_df["url"].to_numpy(),
            "trading_date": batch_df["trading_date"].to_numpy(),
            "category": batch_df["category"].to_numpy(),
            "sentiment_score": prob_positive - prob_negative,
            "sentiment_label": [mapping[int(value)] for value in labels],
            "prob_positive": prob_positive,
            "prob_negative": prob_negative,
            "prob_neutral": prob_neutral,
        }
    )


def validate_classifier_checkpoint(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint missing config.json: {model_dir}")

    config = AutoConfig.from_pretrained(str(model_dir))
    architectures = {value.lower() for value in getattr(config, "architectures", [])}
    if not any("sequenceclassification" in value for value in architectures):
        raise ValueError(f"Checkpoint is not a sequence-classification model: {model_dir}")
    if getattr(config, "num_labels", None) != 3:
        raise ValueError(f"Classifier checkpoint must expose 3 labels: {model_dir}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {model_dir}")
    validate_classifier_checkpoint(model_dir)

    df = read_parquet_table(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = TFRobertaForSequenceClassification.from_pretrained(str(model_dir))

    output_path = ensure_parent_dir(args.output_file)
    checkpoint_path = output_path.with_suffix(".checkpoint.parquet")
    batches: list[pd.DataFrame] = []
    start_offset = 0
    if checkpoint_path.exists():
        checkpoint_df = read_parquet_table(checkpoint_path)
        if not checkpoint_df.empty:
            batches.append(checkpoint_df)
            start_offset = len(checkpoint_df)
            logger.info("Resuming from checkpoint with %d rows", start_offset)

    for start in range(start_offset, len(df), args.batch_size):
        batch_df = df.iloc[start : start + args.batch_size].copy()
        probabilities = predict_probabilities(
            batch_df["input_text_segmented"].astype(str).tolist(),
            model=model,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        batches.append(build_output_rows(batch_df, probabilities))
        processed = start + len(batch_df)
        if processed % args.checkpoint_every == 0:
            pd.concat(batches, ignore_index=True).to_parquet(checkpoint_path, index=False)
            logger.info("Checkpointed %d rows -> %s", processed, checkpoint_path)

    output_df = pd.concat(batches, ignore_index=True)
    output_df.to_parquet(output_path, index=False)
    logger.info("Saved %d inference rows -> %s", len(output_df), output_path)


if __name__ == "__main__":
    main()
