"""TensorFlow script to fine-tune vinai/phobert-base on ViFiC-93M using MLM."""

from __future__ import annotations

import argparse
import logging
import os
import numpy as np
import tensorflow as tf
import tf_keras as keras
from transformers import AutoTokenizer, TFRobertaForMaskedLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT on ViFiC using MLM in TensorFlow.")
    parser.add_argument("--train-file", default="data/fine-tunes/ViFiC-93M/train.txt", help="Path to training sentences")
    parser.add_argument("--val-file", default="data/fine-tunes/ViFiC-93M/val.txt", help="Path to validation sentences")
    parser.add_argument("--output-dir", default="data/models/phobert-financial-mlm", help="Output model directory")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of sentences to load for training")
    parser.add_argument("--val-sample-size", type=int, default=5000, help="Number of sentences to load for validation")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per TPU core / GPU device")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    return parser.parse_args()


def load_sentences(file_path: str, limit: int) -> list[str]:
    """Load sentences from plain-text file up to limit."""
    sentences = []
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return sentences
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
                if len(sentences) >= limit:
                    break
    logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
    return sentences


def mask_tokens(inputs: np.ndarray, tokenizer: AutoTokenizer, mlm_probability: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Prepare masked tokens and labels for Masked Language Modeling in NumPy."""
    labels = np.copy(inputs)
    # Masking probability matrix
    probability_matrix = np.full(labels.shape, mlm_probability)
    
    # Exclude special tokens from masking
    special_tokens_mask = []
    for val in labels.tolist():
        special_tokens_mask.append(
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        )
    special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
    probability_matrix[special_tokens_mask] = 0.0
    
    # Generate mask
    masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
    
    # We only compute loss on masked tokens. Non-masked labels should be set to -100
    labels[~masked_indices] = -100
    
    # 80% of the time, replace masked input tokens with mask_token_id
    indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id
    
    # 10% of the time, replace masked input tokens with a random word
    indices_random = np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
    random_words = np.random.randint(low=0, high=len(tokenizer), size=labels.shape)
    inputs[indices_random] = random_words[indices_random]
    
    # The remaining 10% of the time, keep the token unchanged (but still masked in labels)
    return inputs, labels


def main() -> None:
    args = parse_args()
    
    # --- TPU / GPU Strategy Detection ---
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        logger.info(f"TPU detected: {resolver.master()}")
    except ValueError:
        # Check for GPU
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            logger.info(f"GPU detected, using OneDeviceStrategy: {gpus}")
        else:
            strategy = tf.distribute.get_strategy()
            logger.info("No TPU/GPU detected. Using default CPU strategy.")
            
    # --- Load and Tokenize Data ---
    logger.info("Loading dataset...")
    train_sentences = load_sentences(args.train_file, args.sample_size)
    val_sentences = load_sentences(args.val_file, args.val_sample_size)
    
    if not train_sentences:
        logger.error("No training data found. Exiting.")
        return
        
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    logger.info("Tokenizing sentences...")
    train_encodings = tokenizer(
        train_sentences,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="np"
    )
    val_encodings = tokenizer(
        val_sentences,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="np"
    ) if val_sentences else None

    # --- Mask Tokens for MLM ---
    logger.info("Preparing Masked Language Modeling inputs...")
    train_inputs, train_labels = mask_tokens(train_encodings["input_ids"], tokenizer)
    train_attention = train_encodings["attention_mask"]
    
    if val_encodings:
        val_inputs, val_labels = mask_tokens(val_encodings["input_ids"], tokenizer)
        val_attention = val_encodings["attention_mask"]
        
    # --- Build tf.data.Dataset ---
    logger.info("Building TensorFlow datasets...")
    train_features = {
        "input_ids": train_inputs,
        "attention_mask": train_attention,
        "labels": train_labels
    }
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_features,
        train_labels  # Dummy target, but required for model.fit structure
    )).shuffle(1000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    if val_sentences:
        val_features = {
            "input_ids": val_inputs,
            "attention_mask": val_attention,
            "labels": val_labels
        }
        val_dataset = tf.data.Dataset.from_tensor_slices((
            val_features,
            val_labels
        )).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_dataset = None

    # --- Build and Train Model ---
    logger.info("Initializing PhoBERT model inside distribution strategy...")
    with strategy.scope():
        # Load HuggingFace PyTorch weights into TensorFlow model
        model = TFRobertaForMaskedLM.from_pretrained("vinai/phobert-base", from_pt=True)
        
        # Use tf_keras optimizer to avoid Keras 3 mismatch
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        
        # TF models compute their own loss internally when labels are in the inputs
        model.compile(optimizer=optimizer)

    logger.info("Starting training...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ] if val_dataset else []
    
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    logger.info(f"Saving fine-tuned model weights to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
