# News Sentiment and VN-Index Volatility

This repo uses one local CLI-first sentiment pipeline for article-level classifier training and CafeF inference. ViFiC-specific preparation, silver-label generation, and separate notebook-owned logic are no longer part of the supported contract.

## Workflow

1. Build the upstream processed artifacts:

```bash
python -m src.preprocessing.pipeline \
  --raw-news data/raw/news_VN_cafef.csv \
  --prices data/raw/prices_VN.csv \
  --out-dir data/main/processed
```

2. Prepare a normalized training corpus with the required article-level schema:

```bash
python -m src.sentiment.prepare_training_data \
  --input-file data/main/cafef/training_input.parquet \
  --output-file data/main/cafef/training_corpus.parquet
```

Required training columns:

- `article_id`
- `source`
- `category`
- `published_at`
- `title`
- `body_text`
- `input_text`
- `input_text_segmented`

The labeled corpus adds:

- `label`
- `split`

3. Export a review sample, annotate it outside the repo, and merge the reviewed labels:

```bash
python -m src.sentiment.sample_annotation \
  --input-file data/main/cafef/training_corpus.parquet \
  --output-file data/main/cafef/annotation_sample.csv

python -m src.sentiment.merge_annotations \
  --corpus-file data/main/cafef/training_corpus.parquet \
  --annotations-file data/main/cafef/training_annotations.csv \
  --output-file data/main/cafef/training_labeled.parquet
```

4. Train a 3-label PhoBERT-compatible classifier checkpoint:

```bash
python -m src.sentiment.train_classifier \
  --labeled-input data/main/cafef/training_labeled.parquet \
  --output-dir models/phobert-sentiment/latest
```

The saved checkpoint must be a sequence-classification model with `num_labels == 3`.

5. Prepare CafeF inference input, run inference, and validate the output:

```bash
python -m src.sentiment.run_pipeline \
  --mode infer \
  --model-dir models/phobert-sentiment/latest
```

This inference flow now also writes `data/main/processed/modeling_ready.parquet`, the merged volatility-modeling frame with price features, news intensity features, and inferred sentiment features.

6. Orchestrate end to end from the same CLI:

```bash
python -m src.sentiment.run_pipeline \
  --mode train \
  --training-input data/main/cafef/training_input.parquet \
  --labeled-input data/main/cafef/training_annotations.csv \
  --model-dir models/phobert-sentiment/latest

python -m src.sentiment.run_pipeline \
  --mode full \
  --training-input data/main/cafef/training_input.parquet \
  --labeled-input data/main/cafef/training_annotations.csv \
  --model-dir models/phobert-sentiment/latest
```

## Sentiment Contract

- `src.sentiment.prepare_inputs` prepares unlabeled CafeF inference rows from `data/main/processed/articles_clean.parquet`.
- `src.sentiment.infer_cafef` resolves `--model-dir` first, otherwise defaults to `models/phobert-sentiment/latest`.
- `src.sentiment.validate_inference` rejects stale 4-column `article_sentiment_scores.parquet` outputs and requires:
  - `url`
  - `trading_date`
  - `category`
  - `sentiment_score`
  - `sentiment_label`
  - `prob_positive`
  - `prob_negative`
  - `prob_neutral`
- Operators must regenerate old article-level sentiment outputs with the new classifier. The repo does not auto-upgrade stale inference artifacts.

## Main Artifacts

- `data/main/processed/articles_clean.parquet`
- `data/main/processed/daily_news_prices.parquet`
- `data/main/cafef/cafef_input.parquet`
- `data/main/processed/article_sentiment_scores.parquet`
- `data/main/processed/modeling_ready.parquet`
- `data/main/processed/sentiment_inference_validation.json`
- `data/main/processed/daily_aggregation_validation.json`
- `models/phobert-sentiment/latest/`

## Modeling

Volatility scripts should use `data/main/processed/modeling_ready.parquet` as the default modeling input after sentiment inference. If that artifact is missing, the modeling CLIs can still rebuild the frame from prices, `daily_news_prices.parquet`, and `article_sentiment_scores.parquet`.
