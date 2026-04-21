# GEMINI.md - Instructional Context

This file serves as the foundational instructional context for the **News Sentiments Analysis** project. It provides an overview of the project's purpose, architecture, and development guidelines to ensure consistent and effective AI assistance.

## Project Overview

**News Sentiments Analysis** is a project designed to analyze the sentiment of news articles and headlines. The goal is to provide insights into media trends, public perception, and emotional tone across various news sources.

### Core Objectives
1.  **Data Acquisition:** Fetch news data from APIs (e.g., NewsAPI, GNews) or through web scraping.
2.  **Sentiment Processing:** Utilize Natural Language Processing (NLP) models to classify news content into positive, negative, or neutral sentiments.
3.  **Analysis & Visualization:** Aggregate sentiment scores and visualize trends over time or by topic/source.

### Technical Stack (Anticipated)
- **Language:** Python 3.x
- **NLP Frameworks:** `HuggingFace Transformers`, `NLTK`, `VADER`, or `TextBlob`.
- **Data Handling:** `pandas`, `numpy`.
- **Visualization:** `Plotly`, `Matplotlib`, or `Seaborn`.
- **API Integration:** `requests` for HTTP calls.

## Building and Running

*This project is currently in the initialization phase. The following commands are standard recommendations for this project type.*

### Setup
1.  **Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Dependencies:**
    ```bash
    # Create a requirements.txt if it doesn't exist
    pip install transformers torch pandas scikit-learn requests
    ```

### Execution
- **Main Entry Point:** `python src/main.py` (TODO: Define script)
- **Data Collection:** `python src/ingestion.py` (TODO: Define script)

## Development Conventions

### Architecture
The project should follow a modular structure:
- `src/`: Core logic including ingestion, processing, and analysis modules.
- `data/`: Local storage for cached news items and processed results (ensure `.gitignore` excludes large datasets).
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA).
- `tests/`: Comprehensive test suite using `pytest`.

### Coding Standards
- **Naming:** Follow PEP 8 (snake_case for variables/functions, PascalCase for classes).
- **Types:** Use Python type hints for clarity and robust development.
- **Documentation:** Use Google-style docstrings for all public modules and functions.
- **Commit Messages:** Use descriptive, imperative-style commit messages (e.g., "Add sentiment analysis module").

## Next Steps
- [ ] Initialize Git repository.
- [ ] Create `requirements.txt`.
- [ ] Implement a basic news ingestion script.
- [ ] Set up a baseline sentiment analysis model.
