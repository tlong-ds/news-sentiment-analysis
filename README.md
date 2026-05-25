# News Sentiment and VN-Index Volatility

This repo now treats CafeF as the only supported news corpus in the sentiment workflow.

## Workflow

1. Build `data/main/processed/articles_clean.parquet` and `data/main/processed/daily_news_prices.parquet`:

```bash
python -m src.preprocessing.pipeline \
  --raw-news data/raw/news_VN_cafef.csv \
  --prices data/raw/prices_VN.csv \
  --out-dir data/main/processed
```

2. Prepare article-level inference inputs:

```bash
python -m src.sentiment.prepare_inputs
```

3. Run classifier inference with an explicit fine-tuned checkpoint:

```bash
python -m src.sentiment.infer_cafef \
  --model-dir models/phobert-sentiment-cafef
```

4. Validate the inference artifact before modeling:

```bash
python -m src.sentiment.validate_inference --fail-on-validation
python -m src.sentiment.validate_daily_aggregation
```

## Current sentiment contract

- `src.sentiment.prepare_inputs` reads only `data/main/processed/articles_clean.parquet`.
- `src.sentiment.infer_cafef` reads only `data/main/cafef/cafef_input.parquet`.
- Validation reads only parquet artifacts under `data/main/processed/`.
- No in-repo annotation, silver-label, or domain-adaptation path remains in this checkout.

## Main artifacts

- `data/main/processed/articles_clean.parquet`
- `data/main/processed/daily_news_prices.parquet`
- `data/main/cafef/cafef_input.parquet`
- `data/main/processed/article_sentiment_scores.parquet`

## Modeling

The modeling path is unchanged: article-level sentiment scores are aggregated into daily features and merged with the price/news frame for the hybrid GARCH plus LSTM experiment.
- `garch_std_resid`
- `hybrid_residual_target`

These are the core series used in the pure baseline and hybrid comparison.
