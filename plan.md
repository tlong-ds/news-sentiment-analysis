Let me think through this carefully before building the plan — ASEAN data collection has some real practical constraints worth mapping first.Good — I now have enough to build a precise, actionable data collection plan. Let me synthesize everything.

Several things are now confirmed from the research:

- LSEG MRN News Analytics provides structured analytics including asset sentiment, relevance, and novelty, with history back to 2003 for research and back-testing
- The TRNA (Thomson Reuters News Analytics) feed provides `sentimentClass`, `relevance` (0–1), novelty counts, and volume counts per asset — all directly accessible via the MRN domain
- The standard Eikon Data API is limited to 15 months of news history; deeper historical access requires a separate news archive product through the LSEG account team
- The premium MRN option includes analytics from LSEG's NLP applications including readership and sentiment, delivered as structured feeds with historical archives dating back to 1996

This means there is a **two-tier access problem** you need to resolve immediately before any other work begins. Here is the full data collection plan:

---

## Data Collection Plan: GARCH-MIDAS Sentiment for ASEAN Volatility

---

### Phase 0: Access Verification (Do This First — Week 1)

This is the most critical step and must happen before you scope the sample period or asset universe. Everything else depends on what your LSEG account actually grants.

**Step 0.1 — Clarify your LSEG access tier**

Log into LSEG Workspace (the successor to Eikon) and check which products are activated. The key distinction is:

| Product | What you get | History depth |
|---|---|---|
| Standard Eikon Data API | News headlines + story text only | ~15 months |
| MRN Real-Time News (STORY) | Full news text + metadata, no pre-scored sentiment | Back to 2003 |
| **MRN News Analytics (TRNA)** | **Pre-scored sentiment, relevance, novelty, volume** | **Back to 2003** |
| News Archive | Bulk historical delivery, offline | Back to 1996 |

You need **TRNA specifically** for this paper. It is the product that provides the pre-scored sentiment fields equivalent to RPNA's ESS/CSS. If your university license only covers the standard Eikon Data API, you will only get headline text and must run your own NLP (FinBERT or similar) on it — that is still viable but adds a methodological step and changes the contribution framing slightly.

**Step 0.2 — Test access in Python immediately**

Install the LSEG Data Library and run this diagnostic:

```python
import lseg.data as ld

ld.open_session()

# Test 1: Can you pull daily index prices?
df_prices = ld.get_history(
    universe=["^VNINDEX", ".SETI", ".KLSE", "^JKSE", ".STI", "PSEi.PS"],
    fields=["TRDPRC_1", "ACVOL_UNS"],
    start="2015-01-01",
    end="2024-12-31"
)
print(df_prices.head())

# Test 2: Can you pull news with TRNA sentiment scores?
# This only works if you have TRNA access
headlines = ld.news.get_headlines(
    query="Vietnam economy",
    count=10,
    date_from="2020-01-01",
    date_to="2020-01-31"
)
print(headlines)

# Test 3: Can you retrieve a story with sentiment metadata?
# story_id from above headline result
story = ld.news.get_story(story_id=headlines["storyId"].iloc[0])
print(story)
```

The presence or absence of sentiment fields in the story output immediately tells you whether TRNA is activated on your account. If `sentimentClass`, `relevance`, and `noveltyCount` fields appear in the JSON response, you have full TRNA access.

**Step 0.3 — If TRNA is not activated**

Contact your university's LSEG account manager (your institution has one — find them through the Eikon Help menu or your library's data services desk). Explain you are conducting academic research on news analytics for ASEAN equity markets. Academic TRNA licenses are routinely granted to university researchers at no additional cost because LSEG runs a dedicated academic program. Request specifically: **MRN TRNA historical archive, 2010–2024, ASEAN country codes and equities**. This negotiation should take 1–2 weeks.

---

### Phase 1: Market Data Collection — ASEAN Equity Indices

**Target universe (6 countries):**

| Country | Index | LSEG RIC | Notes |
|---|---|---|---|
| Vietnam | VN-Index | `^VNINDEX` | Ho Chi Minh Stock Exchange |
| Thailand | SET Index | `.SETI` | Or SET50 `.SET50` for large-cap focus |
| Indonesia | IDX Composite (JCI) | `^JKSE` | Jakarta Composite |
| Malaysia | FTSE Bursa KLCI | `.KLSE` | Kuala Lumpur Composite |
| Singapore | STI | `.STI` | Straits Times Index |
| Philippines | PSEi | `PSEi.PS` | Philippine Stock Exchange Index |

**Sample period target:** January 2010 – December 2024. This gives you ~15 years, spanning the Eurozone crisis spillovers, taper tantrum (2013), ASEAN commodity shock (2015–16), COVID-19 (2020), and post-COVID normalization — sufficient regime variation for meaningful out-of-sample evaluation.

**Variables to pull for each index:**

```python
import lseg.data as ld
import pandas as pd

ld.open_session()

ASEAN_RICS = {
    "VN":  "^VNINDEX",
    "TH":  ".SETI",
    "ID":  "^JKSE",
    "MY":  ".KLSE",
    "SG":  ".STI",
    "PH":  "PSEi.PS"
}

fields = [
    "TRDPRC_1",   # closing price
    "OPEN_PRC",   # opening price
    "HIGH_1",     # daily high
    "LOW_1",      # daily low
    "ACVOL_UNS",  # volume (unadjusted)
]

for country, ric in ASEAN_RICS.items():
    df = ld.get_history(
        universe=ric,
        fields=fields,
        start="2010-01-01",
        end="2024-12-31",
        interval="daily"
    )
    df.to_csv(f"data/prices_{country}.csv")
    print(f"{country}: {len(df)} observations")
```

From the closing prices, compute:
- Daily log-returns: $r_t = \log(P_t / P_{t-1})$
- Squared returns $r_t^2$ as the variance proxy for QLIKE evaluation
- Basic return statistics per country to check for data integrity (no spurious zero-return days, trading calendar alignment)

**Important data cleaning issues for ASEAN specifically:**
- Vietnam has circuit breakers (±7% daily limit) — flag these observations, do not remove them (they are genuine volatility events)
- Indonesian and Philippine markets have more frequent exchange closures than developed markets — build a proper trading calendar for each country
- Currency denomination: all indices are local-currency — you do not need to convert, since you are modeling volatility of local returns, not USD-denominated returns

---

### Phase 2: MRN Sentiment Data Collection

This is the core and most complex piece. There are two distinct news series you need to construct, paralleling Bodilsen & Lunde's two-index design.

#### Series A: Country-Level Macro Sentiment Index

This parallels RPNA's GMNA (Global Macro News Analytics) in Bodilsen & Lunde. You want news tagged to each ASEAN country's economy.

**TRNA field mapping to Bodilsen & Lunde equivalents:**

| Bodilsen & Lunde (RPNA) | LSEG TRNA equivalent | Field name |
|---|---|---|
| ESS (Event Sentiment Score) | `sentimentScore` (−1 to +1) or `sentimentClass` | `analytics.sentimentScore` |
| Relevance score (0–100) | `relevance` (0–1) | `analytics.relevance` |
| Novelty score (100 = first story) | `noveltyCount` within time windows | `analytics.noveltyCounts` |
| Country = "US", topic = "economy" | Country topic codes for each ASEAN country | MRN topic/subject codes |

**MRN topic codes for ASEAN macro news** — these are the subject/topic codes you filter on:

| Country | MRN topic code | Description |
|---|---|---|
| Vietnam | `VNM` or `OVIET` | Vietnam economy news |
| Thailand | `THA` or `OTHAI` | Thailand economy |
| Indonesia | `IDN` or `OINDO` | Indonesia economy |
| Malaysia | `MYS` or `OMALA` | Malaysia economy |
| Singapore | `SGP` or `OSING` | Singapore economy |
| Philippines | `PHL` or `OPHIL` | Philippines economy |

You additionally want to combine with topic codes for macro categories: `ECON` (economics), `MKTS` (markets), `MONET` (monetary policy). The exact available codes depend on your TRNA version — verify them in the LSEG topic code reference on the developer portal.

**Python collection loop for macro sentiment:**

```python
import lseg.data as ld
import pandas as pd
import json
from datetime import datetime, timedelta

COUNTRY_CODES = {
    "VN": ["VNM", "OVIET", "ECON"],
    "TH": ["THA", "OTHAI", "ECON"],
    "ID": ["IDN", "OINDO", "ECON"],
    "MY": ["MYS", "OMALA", "ECON"],
    "SG": ["SGP", "OSING", "ECON"],
    "PH": ["PHL", "OPHIL", "ECON"],
}

def fetch_macro_news_batch(country_code, topic_codes, start_date, end_date):
    """
    Fetch TRNA-scored macro news for one country over a date range.
    Returns a DataFrame with timestamp, sentimentScore, relevance, noveltyCount.
    """
    query = " OR ".join([f"T:{tc}" for tc in topic_codes])
    # Add language filter — English only, consistent with Bodilsen & Lunde
    query += " AND LEN:EN"
    
    headlines = ld.news.get_headlines(
        query=query,
        count=10000,  # max per batch
        date_from=start_date,
        date_to=end_date
    )
    
    records = []
    for _, row in headlines.iterrows():
        story = ld.news.get_story(story_id=row["storyId"])
        # Parse TRNA sentiment fields from story metadata
        if story and "analytics" in story:
            records.append({
                "date": row["versionCreated"],
                "storyId": row["storyId"],
                "sentimentScore": story["analytics"].get("sentimentScore"),
                "sentimentClass": story["analytics"].get("sentimentClass"),
                "relevance": story["analytics"].get("relevance"),
                "noveltyCount_1d": story["analytics"].get("noveltyCounts", [{}])[0].get("count"),
                "headline": row.get("text", "")
            })
    
    return pd.DataFrame(records)
```

**Rate limit management:** The LSEG API has request throttling. For 15 years × 6 countries, do not try to pull all at once. Structure as monthly batches with a `time.sleep(0.5)` between requests, and checkpoint to CSV after each month. Budget approximately 2–3 days of runtime for the full macro news pull.

#### Series B: Global Macro Sentiment (Extension beyond Bodilsen & Lunde)

This is your unique ASEAN contribution — testing whether US Fed news and China macro news predict ASEAN volatility beyond domestic macro sentiment. Pull separately:

```python
GLOBAL_CODES = {
    "US_FED": ["FED", "ECON", "USA"],        # Federal Reserve, US economy
    "CHINA":  ["CHN", "OCHIN", "ECON"],       # China macro
    "GLOBAL": ["GLOB", "MKTS"],               # Global market news
}
```

#### Series C: Firm-Level Sentiment (Optional, for extension)

If you want to include firm-specific news (analogous to Bodilsen & Lunde's ENA database), you need the constituent stocks of each index, not just the index itself. This significantly expands the data volume. For a first paper, I recommend starting with **index-level macro sentiment only** and flagging firm-level as future work. This keeps the paper clean and the data collection tractable.

---

### Phase 3: Benchmark and Control Variables

These are all straightforward to pull from LSEG Eikon.

**Uncertainty and volatility indices for benchmark comparison (replicating Bodilsen & Lunde Section 4.6):**

```python
BENCHMARK_RICS = {
    "VIX":    ".VIX",           # CBOE VIX
    "VXEEM":  ".VXEEM",         # EM Volatility Index (more relevant than VIX for ASEAN)
    "EPU":    None,             # Baker, Bloom & Davis — download from policyuncertainty.com directly
}

# Also pull US and China macro conditions indices
# ADS index: download from Philadelphia Fed website
# EPU: download from policyuncertainty.com (free, country-level available for most ASEAN countries)
```

**Country-level EPU indices** — Baker, Bloom & Davis maintain country-level Economic Policy Uncertainty indices for Thailand, Singapore, and a broader Asian EPU. Download from `policyuncertainty.com`. These are free and give you a locally-relevant uncertainty benchmark beyond VIX.

**Exchange rates** — pull USD/local currency for each country as a control variable, since exchange rate volatility is closely linked to equity volatility in ASEAN:

```python
FX_RICS = {
    "VN": "VND=",
    "TH": "THB=",
    "ID": "IDR=",
    "MY": "MYR=",
    "SG": "SGD=",
    "PH": "PHP=",
}
```

---

### Phase 4: Data Assembly and Quality Control

Once all raw data is pulled, these are the non-negotiable cleaning steps before any modeling.

**Step 4.1 — Trading calendar alignment**

Each country has a different holiday calendar. Build a unified trading day indicator per country and align all series. A day where the market is closed gets NaN for returns (exclude from estimation) but news sentiment that arrives on that day should be rolled forward to the next trading day — exactly the convention Bodilsen & Lunde use (footnote 3 of the paper: weekend and holiday news is assigned to the next trading day).

```python
import pandas_market_calendars as mcal

EXCHANGE_CALENDARS = {
    "VN": "HOSE",     # Ho Chi Minh Stock Exchange
    "TH": "SET",
    "ID": "IDX",
    "MY": "BURSA",
    "SG": "SGX",
    "PH": "PSE",
}
```

**Step 4.2 — News coverage diagnostics**

Before building any index, run a coverage audit for each country:

```python
# For each country, report:
# - Total news items per year (check for coverage gaps)
# - Fraction with relevance > 0.5 (check signal quality)
# - Fraction with non-neutral sentimentClass (check informativeness)
# - Distribution of sentimentScore values

def coverage_report(df, country):
    print(f"\n=== {country} ===")
    print(f"Total items: {len(df)}")
    print(f"Items per year:\n{df.groupby(df['date'].dt.year).size()}")
    print(f"High-relevance (>0.5): {(df['relevance'] > 0.5).mean():.1%}")
    print(f"Non-neutral sentiment: {(df['sentimentClass'] != 'neutral').mean():.1%}")
    print(f"Sentiment distribution:\n{df['sentimentClass'].value_counts(normalize=True)}")
```

This is critical for Vietnam specifically. Vietnamese-economy news in Reuters/LSEG may be sparse pre-2015. If annual news counts fall below ~200 items for any country-year, that country's macro sentiment index will be unreliable for that period. You may need to trim the sample start date for thin-coverage countries or exclude them from the main analysis and treat them as a robustness subsample.

**Step 4.3 — Merge and construct daily sentiment series**

```python
def build_daily_sentiment_index(raw_news_df, d=100, lambda_val=None):
    """
    Construct the MIDAS-weighted macro sentiment index.
    If lambda_val is None, it will be estimated during model fitting.
    For diagnostic purposes use lambda_val=1 (equal weights).
    
    Returns daily series aligned to trading calendar.
    """
    # Filter: relevance >= 0.5 (equivalent to relevance=100 in RPNA)
    # Filter: noveltyCount_1d == 1 (first story on event — equivalent to novelty=100 in RPNA)
    # Filter: exclude FX-tagged news (equivalent to Bodilsen & Lunde's FX exclusion)
    # Invert sentiment: use negative score so index is positively correlated with volatility
    
    filtered = raw_news_df[
        (raw_news_df["relevance"] >= 0.5) &
        (raw_news_df["noveltyCount_1d"] == 1) &
        (~raw_news_df["topics"].str.contains("FX|FOREX", na=False))
    ].copy()
    
    filtered["sent_neg"] = -filtered["sentimentScore"]
    
    # Group by trading day
    daily = filtered.groupby("trading_date")["sent_neg"].agg(["mean", "count"])
    
    # Apply equal-weight moving average over d days (lambda=1 baseline)
    index_series = daily["mean"].rolling(window=d, min_periods=10).mean()
    
    return index_series
```

**Step 4.4 — Final dataset structure**

The output going into modeling should be one CSV per country with this structure:

| Column | Description |
|---|---|
| `date` | Trading date |
| `r_t` | Daily log-return |
| `r2_t` | Squared log-return (variance proxy) |
| `M_t` | Macro sentiment index (equal-weight, d=100) |
| `M_t_ew22` | Macro sentiment index (equal-weight, d=22) |
| `news_count_macro` | Raw daily count of macro news items |
| `VIX_t` | VIX level |
| `EPU_t` | Country EPU index (if available) |
| `FX_r_t` | FX log-return (USD/local) |
| `trading_day` | 1 if market open, 0 otherwise |

---

### Phase 5: Risk Register and Contingency Plan

These are the realistic failure modes and how to handle each.

**Risk 1: TRNA not activated on your account**
Contingency: Use `lseg.news.get_story()` to pull raw headline text for each story, then apply FinBERT (pre-trained financial sentiment model, freely available on HuggingFace as `ProsusAI/finbert`) to score sentiment yourself. This is actually a second viable contribution framing — "self-constructed NLP sentiment index vs. professional TRNA scores" — but adds 2–3 weeks of compute and validation work.

**Risk 2: Vietnamese news coverage too sparse pre-2015**
Contingency: Start the Vietnam sample in 2015 (when LSEG's Web Edition came online and HOSE coverage improved). Robustness check uses 2015–2024 for all countries to ensure comparability.

**Risk 3: Eikon API rate limits slow the news pull to weeks**
Contingency: Structure the pull as batch jobs (one country × one year per job), run overnight. Alternatively, request a bulk data delivery through your LSEG academic account — this bypasses the API rate limits entirely and delivers the full historical archive as compressed files.

**Risk 4: MRN topic codes don't cleanly separate macro from firm news for ASEAN**
Contingency: Use a keyword-based secondary filter on headline text (e.g., headlines containing "GDP", "inflation", "central bank", "monetary", "growth forecast" → macro; headlines containing company names → firm-specific). Validate against a random sample of 100 stories manually coded.

**Risk 5: Sentiment coverage gaps create unbalanced panel**
Contingency: Treat the dataset as an unbalanced panel. Singapore and Thailand will have the longest and richest coverage (English-language financial centers); Vietnam and Philippines may require a later start date. Report coverage statistics per country transparently in Section 3.

---

### Timeline Summary

| Week | Task |
|---|---|
| 1 | Verify LSEG access tier; run Python diagnostics; contact account manager if TRNA not active |
| 2–3 | Pull all 6 countries' daily price/return data; build trading calendars; validate |
| 3–5 | Pull macro news sentiment (TRNA) for all 6 countries, 2010–2024; checkpoint monthly |
| 5–6 | Pull global sentiment series (US Fed, China, VIX, EPU); merge all datasets |
| 6–7 | Coverage diagnostics; cleaning; construct equal-weight sentiment indices; final dataset |
| 7+ | Begin modeling (GARCH estimation) |

The single most important thing you can do this week is run the Phase 0 diagnostic. Everything else — sample period, country selection, even the methodological design — can be adjusted once you know exactly what your LSEG access actually contains.
