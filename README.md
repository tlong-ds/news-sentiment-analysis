# News Sentiments Analysis (Vietnam)

This project collects business news from VnExpress to analyze news sentiments.

## Project Overview

The objective is to analyze the sentiment of news articles from VnExpress (English edition) to provide insights into media trends and economic sentiment in Vietnam.

## Directory Structure

```text
.
├── src/
│   ├── ingestion/          # Data collection scripts
│   │   └── vnexpress_scraper.py  # Scrapes VnExpress business news
│   ├── utils/              # Shared utility modules
│   │   └── text_utils.py         # HTML cleaning and text processing
│   └── config.py           # Centralized project configuration
├── data/                   # Collected datasets (CSVs)
├── tests/                  # Diagnostic tests
├── plan.md                 # Original project plan
└── GEMINI.md               # Project foundational context
```

## Setup & Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### VnExpress Scraping

To scrape business news for Vietnam:
```bash
python -m src.ingestion.vnexpress_scraper
```

## Configuration

All global settings, including the date range (`START_DATE`, `END_DATE`) and categories, are managed in `src/config.py`.

## Data Dictionary

- `news_VN_vnexpress.csv`: Scraped business news from VnExpress containing:
    - `url`: Article URL
    - `title`: Article title
    - `date`: Publication date
    - `body`: Cleaned article text
