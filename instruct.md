# Comprehensive Workflow Sequence

---

## Phase 0: Pre-Reprocessing Checklist

Before touching any data, verify the following are true:

- Ingestion fix is committed and tested — `published_at` is preserved in the raw output schema for CafeF articles
- `src/preprocessing/news_alignment.py` helper exists and its 3 tests pass
- You have a working Python environment with all dependencies installed
- You have confirmed the PhoBERT sentiment model (fine-tuned on Vietnamese financial text, following Vu et al. 2023 architecture) is ready to run inference — either locally or via a script
- `data/raw/prices_VN.csv` is current and covers 2015–2024
- You have noted the current artifact row counts as a baseline: `articles_clean.parquet` = 85,819 rows, `daily_news_prices.parquet` = 2,497 rows

Do not proceed until all of the above are confirmed. Reprocessing without a ready sentiment model means you will produce a clean artifact and then sit blocked at the same gap again.

---

## Phase 1: Re-Ingest CafeF (If Needed)

This phase is only necessary if the existing `data/raw/news_VN_cafef.csv` was produced before the `published_at` fix. Inspect the file first.

**Step 1.1 — Inspect existing raw ingestion output**

Open `data/raw/news_VN_cafef.csv` and check whether the `published_at` column exists and is populated with intraday timestamps or is missing/null. If it is populated for a representative sample of articles, the existing raw file is usable and you can skip re-ingestion. If it is absent or all-null, proceed.

**Step 1.2 — Re-run ingestion for CafeF only**

Run the ingestion pipeline for the full date window:

```
python -m src.ingestion.pipeline --start 2015-01-01 --end 2024-12-31 --sources cafef
```

Use `--resume` if the scrape is interrupted. Monitor the scrape ledger to confirm coverage is continuous across the date range.

**Step 1.3 — Verify raw output**

After ingestion completes, inspect `data/raw/news_VN_cafef.csv`:
- Row count should be in the same order of magnitude as before (85,000+)
- `published_at` column should be present
- Spot-check 20–30 rows across different years to confirm timestamps are genuine article publication times, not scrape times
- Confirm `source`, `category`, `title`, `date`, `published_at`, `body` are all populated

**Step 1.4 — Document provenance**

Record the ingestion run date, the date window, the row count, and the proportion of rows where `published_at` is non-null vs. null. This becomes a data provenance note in your thesis methodology section. If a large fraction of articles have null `published_at` — which is likely for older CafeF articles that do not expose intraday timestamps — note this explicitly. Those articles will fall back to date-level alignment.

---

## Phase 2: Reprocess — Cleaning and Trading-Day Alignment

**Step 2.1 — Update the notebook to preserve `published_at`**

Open `notebooks/01_data_processing.ipynb`. Locate the section that reads the raw ingestion output and produces `articles_clean.parquet`. Ensure that `published_at` is carried forward into the cleaned artifact as a column alongside `date`. This is the prerequisite for the stricter alignment rule to function.

**Step 2.2 — Replace the alignment call**

Locate the section in the notebook that maps article dates to trading dates. Replace the current calendar-day shift logic with a call to the `news_alignment.py` helper. The helper applies:
- Same-session assignment if `published_at <= 14:45` on a trading day
- Next-session shift if `published_at > 14:45` on a trading day
- Next-session shift for weekends and holidays
- No 7-day cap on forward shifts, so Tết and other long closures are handled correctly

For articles where `published_at` is null, the helper should fall back to date-level alignment. Document this fallback clearly in the notebook as a comment.

**Step 2.3 — Run the full cleaning notebook**

Execute all cells in `notebooks/01_data_processing.ipynb` from top to bottom. Do not run cells selectively. After completion:
- `data/interim/articles_clean.parquet` should contain the same schema as before plus `published_at` and `trading_date`
- Record the new row count
- Record what proportion of rows used intraday alignment vs. date-level fallback — this is a meaningful data quality metric for the methodology section

**Step 2.4 — Diagnose and resolve the row mismatch**

The previous artifact had 2,498 rows in `prices_VN.csv` and 2,497 rows in `daily_news_prices.parquet`. After reprocessing, run a date-index comparison between the two files to identify which date is missing from the daily frame. Confirm whether it is:
- A known processing artifact such as the first row being dropped due to lag computation
- An unmapped non-trading day
- An unexplained drop

Document the finding as a one-line note in the summary and in the thesis. Do not leave it as "likely harmless."

**Step 2.5 — Verify `daily_news_prices.parquet`**

Check the output daily frame:
- Date range should span 2015–2024 continuously across trading days
- `n_articles` distribution should be inspected — flag any single day with an unusually high count as a potential deduplication artifact
- `log_return` should be computed correctly — verify no NaN values except possibly the first row
- `volume` should be non-zero on all trading days

---

## Phase 3: Sentiment Inference

This is the critical missing piece. Do not skip or defer any step here.

**Step 3.1 — Prepare the inference input**

From `data/interim/articles_clean.parquet`, extract the columns needed for sentiment inference: `url`, `trading_date`, `title`, `body_clean`. The inference model will score each article individually. Keep `trading_date` attached to every row so aggregation in the next step is straightforward.

**Step 3.2 — Run PhoBERT inference**

Run the fine-tuned PhoBERT model over every article in `articles_clean.parquet`. For each article, produce:
- `sentiment_score`: a continuous polarity score, ideally in the range [-1, 1] or [0, 1] depending on your model output
- `sentiment_label`: positive, negative, or neutral, inferred from the score using your threshold rule (score > 0.05 → positive, score < -0.05 → negative, otherwise neutral)

Save the output as `data/sentiment/article_sentiment_scores.parquet` with columns: `url`, `trading_date`, `sentiment_score`, `sentiment_label`.

**Step 3.3 — Validate inference output**

Before aggregating, inspect the inference output:
- Total row count should match `articles_clean.parquet` row count — if rows are missing, identify why
- Distribution of `sentiment_score` across the full corpus — check for degenerate outputs such as all scores clustering near zero or all articles classified as neutral
- Distribution of `sentiment_label` — a reasonable expectation for financial news is roughly 40–50% neutral, with negative and positive shares varying by market period
- Spot-check 20–30 articles manually: read the article title and body, compare to the assigned label, flag obvious misclassifications
- Compute per-year label distributions to check for temporal drift in model output — if the model assigns dramatically different sentiment distributions in 2015 vs. 2023, investigate whether this reflects a real market difference or a data quality issue

**Step 3.4 — Document threshold justification**

The thresholds (> 0.05 positive, < -0.05 negative) need a one-line justification in the thesis. Options: cite a precedent from the literature that uses similar thresholds, report the score distribution and show that ±0.05 is a natural inflection point, or acknowledge it as a heuristic and include a robustness check with alternative thresholds (e.g., ±0.10, ±0.15) in an appendix.

**Step 3.5 — Merge scores back onto the cleaned article artifact**

To support category-filtered sentiment features (Phase 4.3) and to make inspection/debugging easier, merge the inferred sentiment columns back into the cleaned article-level parquet:

```bash
python -m src.sentiment.merge_sentiment_with_articles \
  --articles-file data/interim/articles_clean.parquet \
  --sentiment-file data/sentiment/article_sentiment_scores.parquet \
  --output-file data/interim/articles_with_sentiment.parquet \
  --how left
```

This writes `data/interim/articles_with_sentiment.parquet` plus a small merge coverage report JSON.

---

## Phase 4: Build the Model Frame

**Step 4.1 — Run `build_model_frame`**

Call `build_model_frame(...)` in `src/modeling/dataset.py` with:
- `data/raw/prices_VN.csv` as the price input
- `data/interim/daily_news_prices.parquet` as the daily news intensity input
- `data/sentiment/article_sentiment_scores.parquet` as the sentiment input

The function will aggregate article-level sentiment scores to daily features (`mean_sentiment`, `sentiment_std`, `sentiment_volume`, `negative_share`, `neutral_share`, `positive_share`) and merge them with price and news intensity features.

**Step 4.2 — Handle zero-imputation explicitly**

On trading days with zero articles — weekends are already excluded by trading-day alignment, but some trading days may genuinely have no CafeF coverage — the pipeline imputes:
- `n_articles = 0`
- `mean_sentiment = 0`
- `sentiment_std = 0`
- All sentiment shares = 0

Before accepting this, count how many trading days in 2015–2024 have zero articles. If it is a small number (under 5%), zero-imputation is a defensible approximation. If it is large, consider forward-filling the previous day's sentiment instead, or adding a binary `has_news` indicator as a separate feature. Document whichever choice you make.

**Step 4.3 — Compute the additional variables**

Add the following derived variables recommended from the variable set discussion:
- `net_sentiment = positive_share - negative_share`
- `sentiment_surprise = mean_sentiment - rolling_5day_mean_sentiment`
- `macro_sentiment`: mean sentiment of articles in categories `Vĩ mô` and `Kinh tế` only
- `market_sentiment`: mean sentiment of articles in categories `Chứng khoán` and `Thị trường` only

These require joining back to `articles_clean.parquet` by `trading_date` and `category` before aggregation, which means they are best computed before the `build_model_frame` call or added as a preprocessing step inside it.

**Step 4.4 — Final frame inspection**

Inspect the unified daily model frame:
- Row count should match the number of trading days in 2015–2024
- No unexpected NaN values in price columns
- Sentiment columns populated for trading days with articles, zero-imputed or forward-filled for days without
- `target_next_vol` should be the Parkinson volatility of the following trading day, shifted back by one row — verify the last row has NaN for this column since there is no next day

---

## Phase 5: GARCH Baseline

**Step 5.1 — Prepare the baseline run**

Run:
```
python -m src.modeling.run_experiment \
  --prices data/raw/prices_VN.csv \
  --daily-news data/interim/daily_news_prices.parquet \
  --prepare-only
```

This fits the GARCH(1,1) baseline on the full training window and outputs `garch_conditional_vol`, `garch_forecast_vol`, and `garch_std_resid`.

**Step 5.2 — Validate GARCH output**

Before proceeding to the hybrid model:
- Plot conditional variance over time and visually confirm volatility clustering around known shock periods: the 2020 COVID crash, the April 2022 market manipulation shock, and any major policy events
- Check that GARCH parameters satisfy stationarity: `α + β < 1`
- Inspect standardized residuals for remaining ARCH effects using a Ljung-Box test on squared residuals — if significant ARCH effects remain, the GARCH(1,1) specification may need to be extended to GJR-GARCH or EGARCH to capture the leverage effect
- Record baseline forecast performance metrics: MSE, MAE, QLIKE on the test window

**Step 5.3 — Define train/validation/test split**

The split must be strictly temporal. A reasonable split for a 2015–2024 panel is:
- Training: 2015–2021
- Validation: 2022
- Test: 2023–2024

The 2022 validation year is particularly useful because it contains the April 2022 liquidity manipulation shock, which is a high-volatility out-of-sample regime. The GARCH model should be fit on training data only. Do not refit on validation or test data. The LSTM hyperparameters are tuned on validation performance only.

**Step 5.4 — Compute `hybrid_residual_target`**

For each training observation:
```
hybrid_residual_target = target_next_vol - garch_forecast_vol
```

This is the quantity the LSTM will learn to predict. Inspect its distribution — it should be centered near zero with heavier tails on shock days. If it is systematically positive or negative, the GARCH model is biased and you should investigate before training the LSTM on a biased signal.

---

## Phase 6: LSTM Hybrid Model

**Step 6.1 — Build rolling sequences**

Construct input sequences of length L (typically 10–20 trading days) for the LSTM. Each sequence at time t contains:
- `garch_conditional_vol` for days t-L to t-1
- `garch_std_resid` for days t-L to t-1
- `log_return` for days t-L to t-1
- `mean_sentiment` for days t-L to t-1
- `negative_share` for days t-L to t-1
- `n_articles` for days t-L to t-1
- `volume_zscore_21` for days t-L to t-1

The target for each sequence is `hybrid_residual_target` at time t.

**Step 6.2 — Train the LSTM**

Train on the training split only. Use the validation split for early stopping and hyperparameter selection. Key hyperparameters to tune:
- Sequence length L
- Number of LSTM layers and hidden units
- Dropout rate
- Learning rate and batch size

Monitor validation loss at each epoch. Stop training when validation loss stops improving for a patience window of 10–15 epochs.

**Step 6.3 — Produce hybrid forecasts**

For each test observation, the final hybrid forecast is:
```
hybrid_forecast_vol = garch_forecast_vol + lstm_residual_prediction
```

**Step 6.4 — Evaluate and compare**

Evaluate both the GARCH baseline and the hybrid model on the test set using:
- MSE, MAE, QLIKE for point forecast accuracy
- Diebold-Mariano test for statistical significance of the forecast improvement
- Subperiod analysis: compare improvement during the 2022 shock period vs. calm periods to test the regime-sensitivity secondary research question
- Asymmetry analysis: compare model performance on days following high-negative-sentiment vs. high-positive-sentiment days to test the asymmetry secondary research question

---

## Phase 7: Robustness Checks

Run these after the main results are confirmed:

- Alternative sentiment thresholds: ±0.10, ±0.15 instead of ±0.05
- Alternative volatility targets: Garman-Klass volatility instead of Parkinson, as a check on target sensitivity
- LSTM without sentiment features: confirm that sentiment variables contribute incrementally beyond GARCH outputs and return lags alone
- GARCH-X as an intermediate model: add `mean_sentiment` directly as an exogenous variable in the GARCH variance equation, and compare to the full hybrid — this tests whether the nonlinear LSTM stage adds value beyond a linear sentiment-in-GARCH specification
- Rolling window re-estimation: refit GARCH on an expanding window rather than a fixed training set, to check whether results are sensitive to the estimation window

---

## Phase 8: Write-Up Milestones

Complete these in order, not in parallel with modeling:

1. Data chapter: finalize after Phase 4, before any modeling results exist — document sources, alignment rule, imputation decisions, and all row counts
2. Methodology chapter: finalize after Phase 5 Step 5.3 — document GARCH specification, train/test split, and LSTM architecture before results are known
3. Results chapter: write after Phase 6 Step 6.4 — report baseline first, then hybrid, then robustness checks
4. Discussion chapter: write last — interpret findings in light of Vu et al. 2023 (negative sentiment — variance link), Léber and Egyed 2025 (nonlinear sentiment-variance relationship), and the Vietnamese retail market context
