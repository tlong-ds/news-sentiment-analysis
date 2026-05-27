# Data Pipeline Summary

## Objective

This repository builds a daily modeling dataset for the question:

**Can Vietnamese financial news sentiment improve VN-Index volatility forecasting beyond a pure GARCH baseline?**

The pipeline combines news collection, article cleaning, trading-day alignment, sentiment aggregation, and volatility feature engineering for a two-stage hybrid model.

---

## End-to-End Flow

```text
data/raw/news_VN_cafef.csv  (raw CafeF articles)
        |
        v
src/preprocessing/pipeline.py
    - body cleaning & short-article filter
    - trading-day alignment (align_articles_to_trading_day)
    - daily aggregation (aggregate_daily_news)
    - merge with full price calendar
        |
        +---> data/interim/articles_clean.parquet       (article-level)
        +---> data/interim/daily_news_prices.parquet    (daily merged)
        +---> data/interim/preprocessing_diagnostics.json
        |
        v
optional article-level sentiment scores
        |
        +---> data/sentiment/article_sentiment_scores.parquet
        |
        +---> (optional) src/sentiment/merge_sentiment_with_articles.py
        |           +---> data/interim/articles_with_sentiment.parquet
        |
        v
src/modeling/dataset.py  (sentiment aggregation + market features)
        |
        v
src/modeling/run_experiment.py  (GARCH baseline + hybrid LSTM)
```

---

## Stage 0: Raw Dataset Refresh

The preprocessing stage treats `data/raw/news_VN_cafef.csv` as the single raw input dataset.
If you want snapshots, retention, or rollback, handle that outside preprocessing with data
version control or a separate archival workflow.

To refresh the active raw dataset, re-run the CafeF scraper:
   ```bash
   nohup python -m src.ingestion.pipeline \
     --sources cafef \
     --start 2015-01-01 \
     --end 2024-12-31 \
     --resume \
     > logs/cafef_scrape_YYYYMMDD_HHMMSS.log 2>&1 &
   echo $! > logs/cafef_scrape.pid
   ```

---

## Stage 1: Data Ingestion

The ingestion layer lives in [src/ingestion](src/ingestion).

### Sources

- `cafef` — **primary long-run source (2015–2024)**
  - Sitemap-based discovery, HTML parsing.
  - New scraper runs write `published_at` (ISO timestamp) enabling intraday cutoff logic.
  - Old scraper runs (pre-`published_at`) produce date-only rows; alignment falls back gracefully.
- `vnstock` — optional ticker-level supplement
  - Not part of the primary long-run article corpus.
  - Valid for price refreshes (`src/data/fetch_prices_vnstock.py`) or event-time supplements.

### Raw output files

Per-source outputs land in `data/`:

| File | Description |
|------|-------------|
| `data/raw/news_VN_cafef.csv` | Primary CafeF article corpus (CSV_COLUMNS contract) |
| `data/raw/news_VN_vnstock.csv` | Optional ticker-level supplement |

CSV_COLUMNS contract: `url, source, category, title, date, published_at, body`

### Note on `published_at` coverage

- The first full scrape after this workflow change will populate `published_at` for all new articles.
- Older rows in the active raw dataset may still have empty `published_at`; the preprocessing pipeline handles this with date-only fallback.

---

## Stage 2: Cleaning and Trading-Day Alignment

### Production entrypoint

```bash
python -m src.preprocessing.pipeline \
    --raw-news data/raw/news_VN_cafef.csv \
    --prices data/raw/prices_VN.csv \
    --out-dir data/interim
```

The module [src/preprocessing/pipeline.py](src/preprocessing/pipeline.py) owns all core logic:
- Loading raw news (handles both old schema without `published_at` and new schema with it)
- Body cleaning: strip HTML/URLs/emails, normalize whitespace → `body_clean`
- Short-article filter (`body_len < 100` characters)
- Trading-day alignment via [`align_articles_to_trading_day`](src/preprocessing/news_alignment.py)
- Daily aggregation via `aggregate_daily_news`
- Merge to the full price trading-day calendar (zero-fills zero-news days)
- Machine-readable diagnostics export

The notebook [notebooks/01_data_processing.ipynb](notebooks/01_data_processing.ipynb) is a thin execution + inspection layer that calls `build_preprocessed_outputs(...)` and `export_preprocessed_outputs(...)` from the module, then presents alignment diagnostics and exploratory plots.

### Alignment rule

The alignment helper in [src/preprocessing/news_alignment.py](src/preprocessing/news_alignment.py) applies:

- `published_at ≤ 14:45` on a trading day → **same session** (`same_session`)
- `published_at > 14:45` on a trading day → **forward to next session** (`after_close_forward`)
- weekend or holiday with timestamp → **forward to next trading day** (`non_trading_forward`)
- no timestamp (date-only) → date-level logic only (`date_only_same_day` or `date_only_forward`)
- forward shift is **not capped at 7 days** — Tết closures are handled explicitly

### Long-holiday bunching

Tết and other multi-day closures create a real bunching problem. The alignment helper exposes:

- `alignment_reason`
- `calendar_gap_days`
- `after_close_share`
- `non_trading_share`
- `max_calendar_gap_days`

These flow into both `articles_clean.parquet` (article level) and `daily_news_prices.parquet` (daily level).

---

## Processed Output Files

### `data/interim/articles_clean.parquet`

Article-level dataset for sentiment inference.

| Column | Type | Description |
|--------|------|-------------|
| `url` | str | Source article URL |
| `source` | str | `cafef` |
| `category` | str | CafeF topic category |
| `title` | str | Article title |
| `date` | date | Raw published date (YYYY-MM-DD) |
| `published_at` | str | ISO timestamp or `""` when absent |
| `origin_date` | date | Calendar date used for alignment |
| `trading_date` | date | Aligned VN-Index trading session |
| `has_timestamp` | int | 1 if `published_at` was non-null |
| `is_after_close` | int | 1 if article arrived after 14:45 |
| `alignment_reason` | str | One of the alignment reason codes |
| `calendar_gap_days` | int | Days from `origin_date` to `trading_date` |
| `body_clean` | str | Cleaned article body text |
| `body_len` | int | Character count of `body_clean` |

### `data/interim/daily_news_prices.parquet`

Daily trading-day frame — bridge between preprocessing and modeling.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | VN-Index trading session date |
| `close` | float | VN-Index closing price |
| `open` | float | Opening price |
| `high` | float | Daily high |
| `low` | float | Daily low |
| `volume` | int | Trading volume |
| `log_return` | float | Log(close/prev_close) |
| `n_articles` | int | Articles aligned to this session |
| `n_categories` | int | Unique categories in this session |
| `mean_body_len` | float | Mean article body length |
| `after_close_share` | float | Share of articles arriving post-14:45 |
| `non_trading_share` | float | Share of articles from non-trading origins |
| `max_calendar_gap_days` | int | Max alignment gap in this session |

### `data/interim/preprocessing_diagnostics.json`

Machine-readable provenance summary. Fields include:

- `raw_news_path` — path of the raw dataset used for this run
- `raw_cafef_row_count` — rows in the raw file before any filter
- `after_short_filter_row_count` — rows after short-article removal
- `cleaned_article_row_count` — rows in `articles_clean.parquet`
- `processed_daily_row_count` — rows in `daily_news_prices.parquet`
- `published_at_non_null_share` — fraction of articles with intraday timestamps
- `timestamp_based_alignment_share` — same as above (for session-aware alignment)
- `date_only_fallback_share` — fraction using date-only fallback
- `after_close_forward_shifts` — count of post-14:45 forward shifts
- `non_trading_day_forward_shifts` — count of weekend/holiday forward shifts
- `daily_vs_price_explanation` — human-readable explanation of any row count difference

---

## Stage 3: Sentiment Layer

The sentiment layer is expected by [src/modeling/dataset.py](src/modeling/dataset.py).

### Accepted sentiment input

The modeling pipeline can consume either:

- article-level sentiment scores, or
- already aggregated daily sentiment features

Minimum required fields for article-level sentiment input:

- `trading_date` or `date`
- `sentiment_score`

Optional:

- `sentiment_label`

### Daily sentiment aggregation

If article-level scores are provided, the pipeline aggregates them into daily features:

- `mean_sentiment`
- `sentiment_std`
- `sentiment_volume`
- `negative_share`
- `neutral_share`
- `positive_share`

If `sentiment_label` is missing, labels are inferred from the score using simple thresholds:

- score `> 0.05` → positive
- score `< -0.05` → negative
- otherwise → neutral

---

## Stage 4: Market Feature Engineering

The modeling dataset builder computes additional volatility-oriented features from `data/raw/prices_VN.csv`.

Implemented features include:

- `log_return`
- `abs_return`
- `squared_return`
- `parkinson_vol`
- `gk_vol`
- `target_vol`
- `target_next_vol`
- `volume_zscore_21`

### Target definition

The current realized volatility target is built as:

- `target_vol = parkinson_vol`, with fallback to `abs_return`
- `target_next_vol = target_vol.shift(-1)`

So the forecast task is next-day volatility prediction.

---

## Stage 5: Modeling Frame Construction

The function `build_model_frame(...)` in [src/modeling/dataset.py](src/modeling/dataset.py) merges three inputs:

- VN-Index prices
- daily news intensity features
- daily sentiment features

It also fills missing daily controls with neutral defaults, for example:

- zero article count on no-news days
- zero sentiment volume when no sentiment file is present
- zero mean sentiment and sentiment shares when sentiment is absent

The result is a unified daily frame ready for baseline and hybrid forecasting.

---

## Stage 6: GARCH Baseline and Hybrid Preparation

The experiment runner is [src/modeling/run_experiment.py](src/modeling/run_experiment.py).

### Pure baseline

The baseline model is a Gaussian GARCH(1,1), implemented in `fit_garch11_baseline(...)` in [src/modeling/hybrid.py](src/modeling/hybrid.py).

It produces:

- conditional variance estimates
- one-step-ahead volatility forecasts
- standardized residuals

### Hybrid stage

After baseline fitting, the pipeline adds:

- `garch_conditional_vol`
- `garch_forecast_vol`
- `garch_std_resid`
- `hybrid_residual_target`

where:

`hybrid_residual_target = target_next_vol - garch_forecast_vol`

The LSTM then learns this residual correction using rolling sequences built from:

- GARCH outputs
- past returns
- news intensity features
- daily sentiment features

---

## Main Commands

### 1. Build processed datasets

```bash
python -m src.preprocessing.pipeline \
  --raw-news data/raw/news_VN_cafef.csv \
  --prices data/raw/prices_VN.csv \
  --out-dir data/interim
```

Or run interactively via the notebook:

```bash
jupyter notebook notebooks/01_data_processing.ipynb
```

### 2. Validate experiment compatibility

```bash
python -m src.modeling.run_experiment \
  --prices data/raw/prices_VN.csv \
  --daily-news data/interim/daily_news_prices.parquet \
  --prepare-only
```

### 4. Full CafeF re-ingestion (background)

```bash
nohup python -m src.ingestion.pipeline \
  --sources cafef \
  --start 2015-01-01 \
  --end 2024-12-31 \
  --resume \
  > logs/cafef_scrape_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/cafef_scrape.pid
cat logs/cafef_scrape.pid
```

### 5. Full hybrid experiment

```bash
python -m src.modeling.run_experiment \
  --prices data/raw/prices_VN.csv \
  --daily-news data/interim/daily_news_prices.parquet \
  --sentiment data/sentiment/article_sentiment_scores.parquet
```

---

## Current State and Gaps

**In place:**

- Modular preprocessing pipeline (`src/preprocessing/pipeline.py`)
- CafeF raw file as the canonical notebook input
- Alignment and aggregation helpers in `src/preprocessing/news_alignment.py`
- Machine-readable diagnostics in `data/interim/preprocessing_diagnostics.json`
- 27 passing tests covering ingestion, alignment, and preprocessing pipeline
- Ingestion pipeline with `--resume` support for long runs
- GARCH baseline implementation
- LSTM sequence preparation

**What still limits the full experiment:**

- No committed article-level sentiment score file yet
- `tensorflow` may not be installed in the active environment
- The empirical thesis result cannot be answered until real sentiment scores are generated and the hybrid model is trained

---

## Practical Interpretation

The data pipeline is already capable of producing the full experiment frame. The missing operational piece is the sentiment inference output. Once article-level Vietnamese sentiment scores are generated and saved to CSV, the repo can run the full comparison between:

- pure GARCH volatility forecasts
- hybrid GARCH + sentiment-LSTM volatility forecasts
