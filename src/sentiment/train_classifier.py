"""Train a 3-label article-level sentiment classifier."""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MODELS_DATA_DIR
from src.sentiment.common import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    LABELED_REQUIRED_COLUMNS,
    default_model_dir,
    ensure_dir,
    token_stats,
    validate_classifier_checkpoint,
    validate_required_columns,
)
from src.tracking import (
    add_tracking_arguments,
    build_run_tags,
    collect_cli_params,
    configure_tracking,
    git_commit,
    tracking_config_from_args,
)
from src.utils.io import read_parquet_table

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """Dataset for training and evaluating sentiment classification model."""

    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a 3-label article-level sentiment classifier."
    )
    parser.add_argument("--labeled-input", required=True)
    parser.add_argument("--output-dir", default=str(default_model_dir(MODELS_DATA_DIR)))
    parser.add_argument("--base-model", default="vinai/phobert-base-v2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    add_tracking_arguments(parser, include_registry=True)
    return parser.parse_args()


def _prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        df, LABELED_REQUIRED_COLUMNS, dataset_name="labeled sentiment corpus"
    )
    prepared = df.copy()
    prepared["label"] = prepared["label"].astype(str).str.strip().str.lower()
    prepared["split"] = prepared["split"].astype(str).str.strip().str.lower()
    invalid_labels = sorted(set(prepared["label"]) - set(LABEL_TO_ID))
    if invalid_labels:
        raise ValueError(
            f"Invalid labels in labeled sentiment corpus: {invalid_labels}"
        )
    invalid_splits = sorted(set(prepared["split"]) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(
            f"Invalid splits in labeled sentiment corpus: {invalid_splits}"
        )
    return prepared


def _build_dataset(
    df: pd.DataFrame,
    *,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    encodings = tokenizer(
        df["input_text"].astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    dataset = SentimentDataset(dict(encodings), labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(df: pd.DataFrame, probabilities: np.ndarray) -> dict:
    """Compute performance metrics based on probabilities and gold labels."""
    gold = df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    pred = probabilities.argmax(axis=1)
    accuracy = float((gold == pred).mean()) if len(gold) else 0.0
    confusion = pd.crosstab(
        pd.Series(gold).map(ID_TO_LABEL),
        pd.Series(pred).map(ID_TO_LABEL),
        dropna=False,
    )
    metrics = {
        "rows": int(len(df)),
        "accuracy": accuracy,
        "label_distribution": df["label"].value_counts().to_dict(),
        "predicted_distribution": pd.Series(pred)
        .map(ID_TO_LABEL)
        .value_counts()
        .to_dict(),
        "confusion_matrix": confusion.to_dict(orient="index"),
    }
    return metrics


def summarize_split(df: pd.DataFrame, *, skipped_training: bool) -> dict:
    """Summarize the split statistics when training is skipped."""
    return {
        "rows": int(len(df)),
        "accuracy": None if skipped_training else 0.0,
        "label_distribution": df["label"].value_counts().to_dict(),
        "predicted_distribution": {},
        "skipped_training": skipped_training,
    }


def predict_dataset_probabilities(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Predict label probabilities using the model and a DataLoader."""
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            outputs.append(probs)
    if not outputs:
        return np.empty((0, 3), dtype=float)
    return np.concatenate(outputs, axis=0)


def build_training_report(df: pd.DataFrame, evaluation: dict) -> dict:
    """Build a summary training report dictionary."""
    token_counts = df["input_text"].astype(str).str.split().map(len)
    report = {
        "rows": int(len(df)),
        "splits": df["split"].value_counts().to_dict(),
        "labels": df["label"].value_counts().to_dict(),
        "token_stats": token_stats(token_counts),
        "evaluation": evaluation,
    }
    if "source_dataset" in df.columns:
        report["source_datasets"] = df["source_dataset"].value_counts().to_dict()
        report["split_source_datasets"] = pd.crosstab(
            df["split"], df["source_dataset"]
        ).to_dict(orient="index")
    return report


def train_classifier(
    labeled_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> dict:
    """Train the classifier model using the provided data."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prepared = _prepare_labels(labeled_df)
    output_path = ensure_dir(output_dir)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU device: %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=3,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
        )
    except OSError:
        logger.info(
            "PyTorch weights not found for %s; attempting to load from TensorFlow weights with from_tf=True",
            base_model,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=3,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
            from_tf=True,
        )
    model.to(device)

    train_df = prepared[prepared["split"] == "train"].copy().reset_index(drop=True)
    val_df = prepared[prepared["split"] == "val"].copy().reset_index(drop=True)
    test_df = prepared[prepared["split"] == "test"].copy().reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Labeled sentiment corpus must contain non-empty train, val, and test splits."
        )

    train_ds = _build_dataset(
        train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = _build_dataset(
        val_df,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = _build_dataset(
        test_df,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
    )

    if epochs > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in train_ds:
                optimizer.zero_grad()
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(
                "Epoch %d/%d - Loss: %.4f",
                epoch + 1,
                epochs,
                total_loss / len(train_ds),
            )

        test_probs = predict_dataset_probabilities(model, test_ds, device)
        evaluation = {
            "train": compute_metrics(
                train_df, predict_dataset_probabilities(model, train_ds, device)
            ),
            "val": compute_metrics(
                val_df, predict_dataset_probabilities(model, val_ds, device)
            ),
            "test": compute_metrics(test_df, test_probs),
        }
        source_col = (
            "source_dataset" if "source_dataset" in prepared.columns else "source"
        )
        evaluation["test_by_source"] = {
            str(source_name): compute_metrics(group, test_probs[group.index.to_numpy()])
            for source_name, group in test_df.groupby(source_col)
        }
    else:
        evaluation = {
            "train": summarize_split(train_df, skipped_training=True),
            "val": summarize_split(val_df, skipped_training=True),
            "test": summarize_split(test_df, skipped_training=True),
            "test_by_source": {
                str(source_name): summarize_split(group, skipped_training=True)
                for source_name, group in test_df.groupby(
                    "source_dataset"
                    if "source_dataset" in prepared.columns
                    else "source"
                )
            },
        }

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    (output_path / "label_mapping.json").write_text(
        json.dumps(
            {"label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_path / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "training_report.json").write_text(
        json.dumps(
            build_training_report(prepared, evaluation), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    checkpoint_meta = validate_classifier_checkpoint(output_path)
    return {
        "output_dir": str(output_path),
        "evaluation": evaluation,
        "checkpoint": checkpoint_meta,
    }


def main() -> None:
    """CLI entrypoint for classifier training."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    labeled_df = read_parquet_table(args.labeled_input)
    tracking_config = tracking_config_from_args(args)
    tracking = configure_tracking(tracking_config)
    run_name = tracking_config.run_name or "train_classifier"
    with tracking.start_run(
        run_name=run_name,
        tags=build_run_tags(
            stage="train_classifier",
            pipeline_mode="train_classifier",
            source_dataset="cafef",
            base_model=args.base_model,
        ),
    ):
        tracking.log_params(collect_cli_params(args))
        tracking.log_params(
            {
                "invoked_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit(),
                "labeled_rows": len(labeled_df),
            }
        )
        result = train_classifier(
            labeled_df,
            output_dir=args.output_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
        )
        model_dir = Path(result["output_dir"])
        tracking.log_metrics(result["evaluation"])
        tracking.log_artifact(model_dir, artifact_path="classifier_model")
        tracking.log_artifact(model_dir / "evaluation.json")
        tracking.log_artifact(model_dir / "training_report.json")
        if args.mlflow_register_model:
            tracking.register_model(
                model_name=args.mlflow_registered_model_name,
                artifact_path="classifier_model",
                alias=args.mlflow_model_alias,
            )
        logger.info("Saved classifier checkpoint -> %s", result["output_dir"])


if __name__ == "__main__":
    main()
