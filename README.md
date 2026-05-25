# News Sentiment and VN-Index Volatility

This repo is now structured around one research question:

**Can Vietnamese financial news sentiment improve volatility forecasting of the VN-Index beyond a pure GARCH baseline?**

The implemented workflow is a two-stage hybrid design:

1. Fit a pure GARCH-family baseline on VN-Index daily returns.
2. Feed GARCH residual information, baseline volatility forecasts, and daily sentiment features into an LSTM that learns the residual correction.

## Research Workflow

### Stage 0: Build the data

1. **Scrape** Vietnamese business and market news over the 2015-2024 window (CafeF only).
2. **Preprocess** using the modular pipeline — cleans bodies, aligns to trading day (with
   HoSE 14:45 close cutoff when `published_at` is available), aggregates daily controls,
   and merges with the full price calendar:
   ```bash
    python -m src.preprocessing.pipeline \
        --raw-news data/raw/news_VN_cafef.csv \
        --prices data/raw/prices_VN.csv
   ```
3. **Export:**
   - `data/main/processed/articles_clean.parquet` — article-level with alignment diagnostics
   - `data/main/processed/daily_news_prices.parquet` — daily market + news-intensity frame
   - `data/main/processed/preprocessing_diagnostics.json` — machine-readable provenance summary

### Stage 1: Sentiment pipeline

The supervised sentiment path now runs as a dedicated pipeline:

```bash
cp .env.example .env
python -m src.sentiment.prepare_inputs
python -m src.sentiment.sample_vific
python -m src.sentiment.annotate_vific --dry-run
python -m src.sentiment.build_silver_labels
python -m src.sentiment.train_classifier
python -m src.sentiment.infer_cafef
python -m src.sentiment.validate_inference
```

`ViFiC-93M` is often already word-segmented with underscore-joined Vietnamese terms. The prep step now treats ViFiC as pre-segmented by default and only runs `underthesea` on CafeF so both corpora match PhoBERT's expected tokenization scheme.

The final article-level output consumed by modeling must include at least:

- `trading_date` or `date`
- `sentiment_score`

Optional:

- `sentiment_label` with values like `positive`, `neutral`, `negative`
- `category`, `prob_positive`, `prob_negative`, `prob_neutral`

For live annotation runs, set Gemini keys in `.env`:

```bash
GEMINI_API_KEY=your_key
```

If you pass article-level scores into the modeling pipeline, the repo will aggregate them into daily features:

- `mean_sentiment`
- `sentiment_std`
- `sentiment_volume`
- `positive_share`
- `neutral_share`
- `negative_share`

### Stage 2: Volatility forecasting experiment

The new `src/modeling` package builds a single experiment frame with:

- price-derived volatility proxies: `log_return`, `abs_return`, `parkinson_vol`, `gk_vol`
- news-intensity controls: `n_articles`, `n_categories`, `mean_body_len`
- daily sentiment features
- GARCH baseline outputs: conditional volatility, one-step-ahead forecast, standardized residuals

The experiment target is next-day volatility, with the hybrid model learning:

`hybrid_residual_target = realized_next_day_volatility - garch_forecast_volatility`

## Project Layout

```text
.
├── src/
│   ├── ingestion/                  # News collection (cafef, vnstock, vific)
│   ├── preprocessing/
│   │   ├── pipeline.py             # Modular preprocessing entrypoint
│   │   └── news_alignment.py       # Trading-day alignment primitives
│   ├── modeling/
│   │   ├── dataset.py              # Sentiment aggregation + experiment frame
│   │   ├── hybrid.py               # GARCH baseline + LSTM sequence prep
│   │   └── run_experiment.py       # Reproducible CLI runner
│   ├── data/
│   ├── utils/
│   └── config.py
├── notebooks/
│   └── 01_data_processing.ipynb    # Thin execution + inspection layer
├── data/
│   ├── raw/                        # Scraped files from news and vnstock index prices
│   │   ├── news_VN_cafef.csv       # Raw CafeF corpus (primary)
│   │   ├── news_VN_vnstock.csv
│   │   ├── news_scrape_ledger.json
│   │   └── prices_VN.csv
│   ├── fine-tunes/                 # ViFiC text data for PhoBERT domain adaptation
│   │   ├── ViFiC-93M/
│   │   └── ViFiC-120M/
│   └── processed/                  # Cleaned, tokenized, and aligned datasets
│       ├── articles_clean.parquet
│       ├── daily_news_prices.parquet
│       └── preprocessing_diagnostics.json
└── tests/
    ├── test_preprocessing_alignment.py
    ├── test_preprocessing_pipeline.py
    ├── test_modeling_pipeline.py
    └── test_scrape_news.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Commands

### 1. Rebuild cleaned datasets

**Via CLI (recommended for reproducibility):**

```bash
python -m src.preprocessing.pipeline \
  --raw-news data/raw/news_VN_cafef.csv \
  --prices data/raw/prices_VN.csv \
  --out-dir data/main/processed
```

**Via notebook (for interactive inspection):**

- [notebooks/01_data_processing.ipynb](/Users/bunnypro/Projects/news-sentiments-analysis/notebooks/01_data_processing.ipynb)

If LSEG/Workspace access is unavailable, the preferred fallback is `vnstock`
quote history for `VNINDEX`:

```bash
python -m src.data.fetch_prices_vnstock --start 2015-01-01 --end 2024-12-31
```

### 2. Run the full CafeF scrape (background)

```bash
nohup python -m src.ingestion.pipeline \
  --sources cafef \
  --start 2015-01-01 \
  --end 2024-12-31 \
  --resume \
  > logs/cafef_scrape_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/cafef_scrape.pid
```

### 3. Optional MLM adaptation

```bash
python -m src.sentiment.pretrain_mlm
```

### 4. Validate experiment compatibility

```bash
python -m src.modeling.run_experiment \
  --prices data/raw/prices_VN.csv \
  --daily-news data/main/processed/daily_news_prices.parquet \
  --prepare-only
```

### 5. Run the full hybrid experiment

```bash
python -m src.modeling.run_experiment \
  --prices data/raw/prices_VN.csv \
  --daily-news data/main/processed/daily_news_prices.parquet \
  --sentiment data/main/processed/article_sentiment_scores.parquet
```

This writes:

- `data/main/processed/hybrid_experiment_summary.json`

and reports baseline vs hybrid forecast metrics.

## Current Constraints

- The repo now contains the modeling pipeline, but it still needs actual Vietnamese sentiment scores to answer the thesis question empirically.
- The LSTM stage requires `tensorflow`, which is listed in `requirements.txt` but may not already exist in your active environment.
- The current baseline is implemented as a reproducible Gaussian GARCH(1,1). If you want EGARCH, GJR-GARCH, or a formal model-selection sweep, extend `src/modeling/hybrid.py`.
- The current processed artifact in `data/main/processed/articles_clean.parquet` is CafeF-only in this checkout, so claims about full index-level information coverage should be framed carefully.

## Data Contracts

### `data/main/processed/articles_clean.parquet`

Full column contract:

```
url, source, category, title, date, published_at,
origin_date, trading_date, has_timestamp, is_after_close,
alignment_reason, calendar_gap_days, body_clean, body_len
```

The `published_at` column is always present; it is empty (`""`) for articles
scraped before the intraday timestamp was added to the schema.
The `alignment_reason` values are: `same_session`, `date_only_same_day`,
`after_close_forward`, `date_only_forward`, `non_trading_forward`, `unmapped`.

### `data/main/processed/daily_news_prices.parquet`

Full column contract:

```
date, close, open, high, low, volume, log_return,
n_articles, n_categories, mean_body_len,
after_close_share, non_trading_share, max_calendar_gap_days
```

All price trading days are preserved; zero-news days are zero-filled (no NaN).

The `vnstock` source is intended as a ticker-news feed with session-alignable
timestamps when the upstream provider exposes them. It should not be described
as a full editorial corpus, and coverage may be truncated by provider
pagination unless the configured page window is increased.

### Article-level sentiment input

Minimum columns:

- `trading_date` or `date`
- `sentiment_score`

Optional:

- `sentiment_label`

### Experiment output fields

The modeling frame includes:

- `target_vol`
- `target_next_vol`
- `garch_forecast_vol`
- `garch_std_resid`
- `hybrid_residual_target`

These are the core series used in the pure baseline and hybrid comparison.
