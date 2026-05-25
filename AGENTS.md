# Repository Guidelines

## Project Structure & Module Organization

Core code lives in `src/`. The main paths are:
- `src/ingestion/`: CafeF and VNStock scraping entrypoints.
- `src/preprocessing/`: raw-news cleaning, trading-day alignment, and processed artifact export.
- `src/sentiment/`: supervised PhoBERT-style training, bootstrap labeling, inference, and validation.
- `src/modeling/`: dataset assembly and volatility experiments.

Tests live in `tests/`, with fixtures in `tests/fixtures/`. Data artifacts are staged under `data/main/processed/` and `data/main/cafef/`. Notebooks in `notebooks/` are for inspection, not the primary production contract. The report sources live in `report/`.

## Build, Test, and Development Commands

Install dependencies with `pip install -r requirements.txt`.

Key workflows:
- `python -m src.preprocessing.pipeline --raw-news data/raw/news_VN_cafef.csv --prices data/raw/prices_VN.csv --out-dir data/main/processed`
  Builds `articles_clean.parquet` and `daily_news_prices.parquet`.
- `python -m src.sentiment.run_pipeline --mode train --training-input data/main/cafef/training_input.parquet --labeled-input data/main/cafef/training_annotations.csv --model-dir models/phobert-sentiment/latest`
  Prepares training data and trains the classifier.
- `python -m src.sentiment.run_pipeline --mode infer --model-dir models/phobert-sentiment/latest`
  Runs CafeF inference and validation.
- `pytest -q`
  Runs the test suite.
- `python scripts/validate.py`
  Runs style, formatting, custom AST checks, and the full test suite. **All code changes must pass this command.**
- `python scripts/scaffold.py --name <module_name> --type <type>`
  Helper to bootstrap a new pipeline stage module and its unit test.

## Coding Style, Naming Conventions, & Quality Gates

Use Python with 4-space indentation, `snake_case` for functions/variables/files, and short docstrings only where needed. Add type annotations to all new functions. 

Keep modules CLI-friendly: most pipeline stages are runnable with `python -m ...`. Prefer `pandas`/`pyarrow` artifacts over ad hoc CSV rewrites. Match existing artifact names such as `articles_clean.parquet`, `training_corpus.parquet`, and `article_sentiment_scores.parquet`.

Every code change must satisfy:
1. `python scripts/validate.py` must run and exit with 0 (includes Ruff and custom AST rules).
2. No raw `print()` statements in `src/` library code outside of `main()` or CLI blocks unless marked with `# noqa: print`.
3. No imports from `tests/` in `src/`.

## Testing Guidelines

Tests use `pytest`. Name files `tests/test_<area>.py` (or `test_<type>_<name>.py` for scaffolded modules) and test functions `test_<behavior>()`. Add focused fixture-driven tests for schema contracts, label merging, and pipeline smoke paths. When changing `src/sentiment/`, run at least `pytest -q tests/test_sentiment_pipeline.py tests/test_sentiment_prepare_training_data.py`.

## Commit & Pull Request Guidelines

Recent history uses conventional prefixes such as `fix:` and `chore:`. Keep commit subjects short, imperative, and scoped to one change. PRs should state:
- the pipeline stage affected,
- the exact commands run for verification (must include validation run),
- any artifact or schema changes,
- any data or model outputs reviewers should inspect.

## Data & Configuration Notes

Do not treat notebooks as the source of truth for production behavior; update the matching `src/` module first. Preserve the current downstream contract: CafeF remains the inference corpus, and modeling still consumes classifier-generated article sentiment outputs.

