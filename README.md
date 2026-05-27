# News Sentiment and VN-Index Volatility

This repo uses one local CLI-first sentiment pipeline for article-level classifier training and CafeF inference. ViFiC-specific preparation, silver-label generation, and separate notebook-owned logic are no longer part of the supported contract.

## Workflow

1. Build the upstream processed artifacts:

```bash
python -m src.preprocessing.pipeline \
  --raw-news data/raw/news_VN_cafef.csv \
  --prices data/raw/prices_VN.csv \
  --out-dir data/interim
```

2. Prepare a normalized training corpus with the required article-level schema:

```bash
python -m src.sentiment.prepare_training_data \
  --input-file data/interim/training_input.parquet \
  --output-file data/interim/training_corpus.parquet
```

Required training columns:

- `article_id`
- `source`
- `category`
- `published_at`
- `title`
- `body_text`
- `input_text`

The labeled corpus adds:

- `label`
- `split`

3. Export a review sample, annotate it outside the repo, and merge the reviewed labels:

```bash
python -m src.sentiment.sample_annotation \
  --input-file data/interim/training_corpus.parquet \
  --output-file data/interim/annotation_sample.csv

# Merge annotations: if you have manual reviewed annotations, pass them via --annotations-file.
# Otherwise the pipeline will run bootstrap labeling and use its internal output.
python -m src.sentiment.merge_annotations \
  --corpus-file data/interim/training_corpus.parquet \
  --annotations-file data/interim/training_bootstrap_labels.parquet \
  --output-file data/interim/training_labeled.parquet
```

4. Train a 3-label PhoBERT-compatible classifier checkpoint:

```bash
python -m src.sentiment.train_classifier \
  --labeled-input data/interim/training_labeled.parquet \
  --output-dir models/phobert-sentiment/latest
```

The saved checkpoint must be a sequence-classification model with `num_labels == 3`.

5. Prepare CafeF inference input, run inference, and validate the output:

```bash
python -m src.sentiment.run_pipeline \
  --mode infer \
  --model-dir models/phobert-sentiment/latest
```

This inference flow now also writes `data/interim/modeling_ready.parquet`, the merged volatility-modeling frame with price features, news intensity features, and inferred sentiment features.

Optional: merge article-level sentiment back into the cleaned articles parquet for inspection and category-based features:

```bash
python -m src.sentiment.merge_sentiment_with_articles \
  --how left
```

6. Orchestrate end to end from the same CLI:

```bash
python -m src.sentiment.run_pipeline \
  --mode train \
  --training-input data/interim/training_input.parquet \
  --model-dir models/phobert-sentiment/latest

python -m src.sentiment.run_pipeline \
  --mode full \
  --training-input data/interim/training_input.parquet \
  --model-dir models/phobert-sentiment/latest
```

## DVC Pipelines

This repo ships two reproducible DVC pipelines (configured by `params.yaml`):

```bash
dvc repro pipelines/sentiment/dvc.yaml
dvc repro pipelines/volatility/dvc.yaml
```

- `pipelines/sentiment/dvc.yaml`: raw → preprocess → prepare_inputs → infer → (optional merge/validations) → build modeling frame (`modeling_ready.parquet`)
- `pipelines/volatility/dvc.yaml`: `modeling_ready.parquet` → volatility experiments → figures/tables

Notes:
- The inference stage expects a local sentiment checkpoint directory (see `paths.model_dir` in `params.yaml`).
- If `run_robustness` is too slow for iteration, lower `volatility.epochs` in `params.yaml` and re-run:
  `dvc repro pipelines/volatility/dvc.yaml:run_robustness`

## Sentiment Contract

- `src.sentiment.prepare_inputs` prepares unlabeled CafeF inference rows from `data/interim/articles_clean.parquet`.
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

- `data/interim/articles_clean.parquet`
- `data/interim/daily_news_prices.parquet`
- `data/interim/cafef_input.parquet`
- `data/sentiment/article_sentiment_scores.parquet`
- `data/interim/articles_with_sentiment.parquet` (optional)
- `data/interim/modeling_ready.parquet`
- `data/interim/sentiment_inference_validation.json`
- `data/interim/daily_aggregation_validation.json`
- `models/phobert-sentiment/latest/`

## Modeling

## Volatility Modeling Input
Volatility scripts should use `data/interim/modeling_ready.parquet` as the default modeling input after sentiment inference. If that artifact is missing, the modeling CLIs can still rebuild the frame from prices, `daily_news_prices.parquet`, and `article_sentiment_scores.parquet`.
