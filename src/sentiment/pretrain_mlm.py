"""Optional MLM domain adaptation on segmented ViFiC inputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from transformers import AutoTokenizer, TFRobertaForMaskedLM

from src.config import MODELS_DATA_DIR, VIFIC_NORMALIZED_DIR

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional PhoBERT MLM adaptation on ViFiC.")
    parser.add_argument("--input-file", default=f"{VIFIC_NORMALIZED_DIR}/vific_pretraining.parquet")
    parser.add_argument("--text-column", default="input_text_segmented")
    parser.add_argument("--base-model", default="vinai/phobert-base-v2")
    parser.add_argument("--output-dir", default=f"{MODELS_DATA_DIR}/phobert-vific-adapted")
    parser.add_argument("--report-file", default=f"{MODELS_DATA_DIR}/phobert-vific-adapted/pretraining_report.json")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    return parser.parse_args()


def mask_tokens(inputs: np.ndarray, tokenizer: AutoTokenizer, mlm_probability: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    labels = np.copy(inputs)
    probability_matrix = np.full(labels.shape, mlm_probability)
    special_tokens_mask = []
    for row in labels.tolist():
        special_tokens_mask.append(tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True))
    special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
    probability_matrix[special_tokens_mask] = 0.0

    masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
    labels[~masked_indices] = -100

    indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    indices_random = np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
    random_words = np.random.randint(low=0, high=len(tokenizer), size=labels.shape)
    inputs[indices_random] = random_words[indices_random]
    return inputs, labels


def build_dataset(features: dict[str, np.ndarray], labels: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(min(len(labels), 4096))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class MlMValidationCallback(keras.callbacks.Callback):
    def __init__(self, val_dataset: tf.data.Dataset, output_dir: Path, patience: int) -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.patience = patience
        self.best_val_loss = float("inf")
        self.best_perplexity = float("inf")
        self.best_epoch = -1
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        results = self.model.evaluate(self.val_dataset, verbose=0, return_dict=True)
        val_loss = float(results["loss"])
        perplexity = float(np.exp(min(val_loss, 20.0)))
        if logs is not None:
            logs["val_perplexity"] = perplexity
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_perplexity = perplexity
            self.best_epoch = epoch
            self.wait = 0
            self.model.save_pretrained(str(self.output_dir))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df = pd.read_parquet(args.input_file)
    texts = df[args.text_column].dropna().astype(str).tolist()
    if not texts:
        raise ValueError("No pretraining texts were found.")

    gpus = tf.config.list_physical_devices("GPU")
    skip_due_to_compute = len(gpus) == 0

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="np",
    )
    total_rows = len(encodings["input_ids"])
    val_size = max(1, int(total_rows * args.validation_split))
    train_size = max(1, total_rows - val_size)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    train_inputs, val_inputs = input_ids[:train_size], input_ids[train_size:]
    train_attention, val_attention = attention_mask[:train_size], attention_mask[train_size:]
    train_masked, train_labels = mask_tokens(train_inputs.copy(), tokenizer)
    val_masked, val_labels = mask_tokens(val_inputs.copy(), tokenizer)

    train_features = {"input_ids": train_masked, "attention_mask": train_attention, "labels": train_labels}
    val_features = {"input_ids": val_masked, "attention_mask": val_attention, "labels": val_labels}
    train_dataset = build_dataset(train_features, train_labels, args.batch_size, shuffle=True)
    val_dataset = build_dataset(val_features, val_labels, args.batch_size, shuffle=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "input_file": args.input_file,
        "base_model": args.base_model,
        "gpu_detected": bool(gpus),
        "skip_recommended_due_to_compute": skip_due_to_compute,
        "train_rows": int(train_size),
        "val_rows": int(total_rows - train_size),
        "epochs_requested": args.epochs,
    }

    model = TFRobertaForMaskedLM.from_pretrained(args.base_model, from_pt=True)
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer)
    callback = MlMValidationCallback(val_dataset, output_dir, patience=args.patience)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=[callback], verbose=1)

    if callback.best_epoch < 0:
        model.save_pretrained(str(output_dir))
        callback.best_epoch = len(history.history.get("loss", [])) - 1
        callback.best_val_loss = float(history.history.get("val_loss", [0.0])[-1]) if history.history.get("val_loss") else 0.0
        callback.best_perplexity = float(np.exp(min(callback.best_val_loss, 20.0))) if callback.best_val_loss else 0.0

    tokenizer.save_pretrained(str(output_dir))
    report.update(
        {
            "best_epoch": callback.best_epoch,
            "best_val_loss": callback.best_val_loss,
            "best_val_perplexity": callback.best_perplexity,
            "early_stopped": bool(callback.wait >= args.patience),
            "checkpoint_path": str(output_dir),
        }
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved MLM-adapted checkpoint -> %s", output_dir)


if __name__ == "__main__":
    main()
