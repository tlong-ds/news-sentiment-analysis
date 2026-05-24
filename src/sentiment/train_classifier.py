"""Train a supervised PhoBERT sentiment classifier in TensorFlow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, TFRobertaForSequenceClassification, create_optimizer

from src.config import ANNOTATION_DATA_DIR, MODELS_DATA_DIR
from src.sentiment.common import ID_TO_LABEL, LABEL_TO_ID
from src.sentiment.build_silver_labels import compute_cohen_kappa

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised PhoBERT sentiment classifier.")
    parser.add_argument("--dataset-file", default=f"{ANNOTATION_DATA_DIR}/sentiment_labeled_full.csv")
    parser.add_argument("--base-model", default="vinai/phobert-base-v2")
    parser.add_argument("--output-dir", default=f"{MODELS_DATA_DIR}/phobert-sentiment-cafef")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def _encode_split(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    encodings = tokenizer(
        df["input_text_segmented"].astype(str).tolist(),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="np",
    )
    features = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }
    labels = df["label_encoded"].to_numpy(dtype=np.int32)
    return features, labels


class ValidationMetricsCallback(keras.callbacks.Callback):
    def __init__(self, val_features: dict[str, np.ndarray], val_labels: np.ndarray, output_dir: Path) -> None:
        super().__init__()
        self.val_features = val_features
        self.val_labels = val_labels
        self.output_dir = output_dir
        self.best_macro_f1 = -1.0
        self.best_epoch = -1
        self.wait = 0
        self.patience = 3
        self.early_stopped = False

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logits = self.model(self.val_features, training=False).logits.numpy()
        preds = np.argmax(logits, axis=1)
        macro_f1 = float(f1_score(self.val_labels, preds, average="macro"))
        logs = logs or {}
        logs["val_macro_f1"] = macro_f1
        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.best_epoch = epoch
            self.wait = 0
            self.model.save_pretrained(str(self.output_dir))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stopped = True
                self.model.stop_training = True


def _build_dataset(features: dict[str, np.ndarray], labels: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(len(labels))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df = pd.read_csv(args.dataset_file)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_features, train_labels = _encode_split(train_df, tokenizer, args.max_length)
    val_features, val_labels = _encode_split(val_df, tokenizer, args.max_length)
    test_features, test_labels = _encode_split(test_df, tokenizer, args.max_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))

    class_weights_raw = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(ID_TO_LABEL.keys())),
        y=train_labels,
    )
    class_weights = {int(idx): float(weight) for idx, weight in enumerate(class_weights_raw)}

    model = TFRobertaForSequenceClassification.from_pretrained(args.base_model, num_labels=3, from_pt=True)
    steps_per_epoch = max(1, int(np.ceil(len(train_labels) / args.batch_size)))
    total_train_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_train_steps * 0.10))
    optimizer, schedule = create_optimizer(
        init_lr=args.learning_rate,
        num_train_steps=total_train_steps,
        num_warmup_steps=warmup_steps,
        weight_decay_rate=args.weight_decay,
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)

    train_dataset = _build_dataset(train_features, train_labels, args.batch_size, shuffle=True)
    val_dataset = _build_dataset(val_features, val_labels, args.batch_size, shuffle=False)
    metrics_callback = ValidationMetricsCallback(val_features, val_labels, output_dir)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=[metrics_callback],
        verbose=1,
    )

    best_model = TFRobertaForSequenceClassification.from_pretrained(str(output_dir))
    logits = best_model(test_features, training=False).logits.numpy()
    preds = np.argmax(logits, axis=1)
    test_accuracy = float((preds == test_labels).mean())
    test_macro_f1 = float(f1_score(test_labels, preds, average="macro"))
    report = classification_report(
        test_labels,
        preds,
        labels=[0, 1, 2],
        target_names=[ID_TO_LABEL[idx] for idx in [0, 1, 2]],
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "best_epoch": metrics_callback.best_epoch,
        "best_val_macro_f1": metrics_callback.best_macro_f1,
        "early_stopped": metrics_callback.early_stopped,
        "optimizer": "AdamW",
        "warmup_steps": warmup_steps,
        "total_train_steps": total_train_steps,
        "label_to_id": LABEL_TO_ID,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "class_counts": {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
        "classification_report": report,
        "confusion_matrix": confusion_matrix(test_labels, preds, labels=[0, 1, 2]).tolist(),
    }
    if {"llm_a_label", "llm_b_label"}.issubset(test_df.columns):
        summary["llm_inter_model_kappa_test"] = compute_cohen_kappa(
            test_df["llm_a_label"],
            test_df["llm_b_label"],
            list(LABEL_TO_ID.keys()),
        )
    spot_check_path = Path(ANNOTATION_DATA_DIR) / "spot_check_results.json"
    if spot_check_path.exists():
        summary["spot_check_results"] = json.loads(spot_check_path.read_text(encoding="utf-8"))
    metrics_path = output_dir / "evaluation.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved classifier checkpoint and metrics to %s", output_dir)


if __name__ == "__main__":
    main()
