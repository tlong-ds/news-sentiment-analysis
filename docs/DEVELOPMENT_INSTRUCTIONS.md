# Reusable Development Instructions

Welcome to the **News Sentiments Analysis** repository. This document outlines the development conventions, coding standards, and quality gates that must be followed by every developer and AI agent working on this codebase.

---

## 1. Project Structure

Core code is structured as follows:
- `src/ingestion/`: CafeF and VNStock scraping entrypoints and sources.
- `src/preprocessing/`: Raw-news cleaning, trading-day alignment, and processed parquet exports.
- `src/sentiment/`: Supervised PhoBERT-style model fine-tuning, labeling, inference, and validation.
- `src/modeling/`: Dataset assembly and statistical/volatility experiments.
- `src/utils/`: Common project helper functions (date formats, IO, text normalization).
- `tests/`: Comprehensive test suite using `pytest`.

---

## 2. Coding Conventions

- **Indentation**: Use standard Python 4-space indentation.
- **Naming Style**: Use `snake_case` for all functions, variables, modules, and files. Use `PascalCase` for classes.
- **Type Annotations**: Use Python type hints (e.g. `def process_data(inputs: dict) -> dict:`) on all new functions.
- **Docstrings**: Provide Google-style docstrings for all public classes, functions, and modules, describing parameters and return values.

---

## 3. Scaffolding New Modules

When adding a new pipeline stage or module, use the repository's scaffolding script to automatically bootstrap the source and unit test file with standard loggers and structure:

```bash
python scripts/scaffold.py --name <module_name> --type <ingestion|preprocessing|sentiment|modeling>
```

This creates:
1. `src/<type>/<module_name>.py`
2. `tests/test_<type>_<module_name>.py`

---

## 4. Linting and Custom Rules

We enforce standard styling with `ruff` and run custom semantic checks via a custom AST parser.
Run lint checks locally with:

```bash
python scripts/lint.py
```

### Enforced AST Rules:
- **No Raw `print()` in Library Code**: Use the `logging` module or return values instead of raw `print` statements in production source files. In scripts/CLI tools, print is allowed inside the `main()` function or within the `if __name__ == "__main__":` block. If you must use a print elsewhere, add a `# noqa: print` comment to that line to bypass the check.
- **No Import of Tests in `src/`**: Files under `src/` are forbidden from importing any module or helper from the `tests/` directory to prevent circular or test dependencies in production code.
- **Stylistic Warnings**: Missing docstrings on public components are flagged as warnings but do not fail the build.

---

## 5. Verification Gate (Pre-Commit / Pre-Completion)

Every single code change **must** pass the unified validation script before being committed or marked as complete:

```bash
python scripts/validate.py
```

This command runs both the linter (`scripts/lint.py`) and the full pytest suite (`pytest`).

> [!IMPORTANT]
> If any validation fails, the change is considered invalid and must not be committed. Keep all code format-compliant by running `ruff format src tests` before running the validator.
