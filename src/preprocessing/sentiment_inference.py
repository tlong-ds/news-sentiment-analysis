"""TensorFlow script to run zero-shot prompt-based sentiment inference on article titles."""

from __future__ import annotations

import argparse
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, TFRobertaForMaskedLM
import underthesea

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PhoBERT prompt-based sentiment inference.")
    parser.add_argument("--model-dir", default="data/models/phobert-financial-mlm", help="Path to fine-tuned model")
    parser.add_argument("--articles-file", default="data/processed/articles_clean.csv", help="Path to clean articles")
    parser.add_argument("--output-file", default="data/processed/article_sentiment_scores.csv", help="Output scores path")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--max-length", type=int, default=128, help="Max prompt length")
    return parser.parse_args()


def load_model_and_tokenizer(model_dir: str) -> tuple[TFRobertaForMaskedLM, AutoTokenizer]:
    """Load the model and tokenizer, falling back to base model if not fine-tuned yet."""
    tokenizer_name = "vinai/phobert-base"
    
    if os.path.exists(model_dir) and any(os.listdir(model_dir)):
        logger.info(f"Loading fine-tuned model and tokenizer from local folder: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = TFRobertaForMaskedLM.from_pretrained(model_dir)
    else:
        logger.info(f"Local model not found. Loading pre-trained base: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = TFRobertaForMaskedLM.from_pretrained(tokenizer_name, from_pt=True)
        
    return model, tokenizer


def segment_text(text: str) -> str:
    """Normalize and word-segment text for PhoBERT using underthesea."""
    if not text or not isinstance(text, str):
        return ""
    # underthesea word_tokenize with format="text" joins compound syllables with underscores
    return underthesea.word_tokenize(text, format="text")


def main() -> None:
    args = parse_args()
    
    if not os.path.exists(args.articles_file):
        logger.error(f"Articles file not found: {args.articles_file}. Please run the preprocessing pipeline first.")
        return
        
    logger.info("Loading articles...")
    df = pd.read_csv(args.articles_file)
    logger.info(f"Loaded {len(df)} articles.")
    
    # Check for TPU/GPU to speed up inference
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU detected: {gpus}. Running on GPU.")
    else:
        logger.info("Running on CPU.")
        
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    
    # Resolve target sentiment token IDs
    pos_id = tokenizer.convert_tokens_to_ids("tích_cực")
    neg_id = tokenizer.convert_tokens_to_ids("tiêu_cực")
    neu_id = tokenizer.convert_tokens_to_ids("trung_tính")
    
    # Fallback to simple words if compound words are not in tokenizer vocab
    unk_id = tokenizer.unk_token_id
    if pos_id == unk_id or pos_id is None:
        pos_id = tokenizer.convert_tokens_to_ids("tốt")
    if neg_id == unk_id or neg_id is None:
        neg_id = tokenizer.convert_tokens_to_ids("xấu")
    if neu_id == unk_id or neu_id is None:
        neu_id = tokenizer.convert_tokens_to_ids("thường")
        
    logger.info(f"Resolved sentiment token IDs: positive={pos_id}, negative={neg_id}, neutral={neu_id}")
    
    # Pre-segment titles
    logger.info("Word-segmenting article titles...")
    titles_segmented = []
    for title in tqdm(df["title"].astype(str), desc="Segmenting"):
        titles_segmented.append(segment_text(title))
        
    # Construct prompts
    mask_token = tokenizer.mask_token
    prompts = [f"{t} . Đây là tin_tức {mask_token} ." for t in titles_segmented]
    
    # Inference batching
    logger.info("Running sentiment scoring...")
    scores = []
    labels = []
    
    n_batches = int(np.ceil(len(prompts) / args.batch_size))
    for i in tqdm(range(n_batches), desc="Scoring"):
        batch_prompts = prompts[i * args.batch_size : (i + 1) * args.batch_size]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_prompts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="tf"
        )
        
        # Run inference
        outputs = model(encodings)
        logits = outputs.logits.numpy()  # shape: (batch_size, seq_len, vocab_size)
        input_ids = encodings["input_ids"].numpy()
        
        # Extract target probabilities at masked position
        for b_idx in range(len(batch_prompts)):
            # Find the position of the mask token
            mask_positions = np.where(input_ids[b_idx] == tokenizer.mask_token_id)[0]
            if len(mask_positions) > 0:
                mask_pos = mask_positions[0]
            else:
                # Fallback to the last token in the sequence if mask not found
                mask_pos = len(input_ids[b_idx]) - 1
                
            # Get logits at the mask position
            seq_logits = logits[b_idx, mask_pos, :]
            
            # Extract target token logits
            pos_logit = float(seq_logits[pos_id])
            neg_logit = float(seq_logits[neg_id])
            neu_logit = float(seq_logits[neu_id])
            
            # Compute softmax
            val_logits = np.array([pos_logit, neg_logit, neu_logit])
            exp_logits = np.exp(val_logits - np.max(val_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # sentiment_score = prob_positive - prob_negative (range: [-1, 1])
            score = float(probs[0] - probs[1])
            scores.append(score)
            
            # Determine label using thresholds (> 0.05 positive, < -0.05 negative, else neutral)
            if score > 0.05:
                labels.append("positive")
            elif score < -0.05:
                labels.append("negative")
            else:
                labels.append("neutral")
                
    # Save output CSV
    output_df = pd.DataFrame({
        "url": df["url"],
        "trading_date": df["trading_date"],
        "sentiment_score": scores,
        "sentiment_label": labels
    })
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output_df.to_csv(args.output_file, index=False)
    logger.info(f"Sentiment inference finished. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
