"""Fetch VN-Index daily OHLCV data from Yahoo Finance.

This writes the repo's expected `data/prices_VN.csv` schema:

    Date,TRDPRC_1,OPEN_PRC,HIGH_1,LOW_1,ACVOL_UNS

Usage:
    python -m src.data.fetch_prices_yahoo
    python -m src.data.fetch_prices_yahoo --start 2015-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import RAW_DATA_DIR, END_DATE, START_DATE

YAHOO_TICKER = "^VNINDEX.VN"
OUTPUT_NAME = "prices_VN.csv"
OUTPUT_COLUMNS = ["Date", "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch VN-Index daily OHLCV data from Yahoo Finance."
    )
    parser.add_argument("--ticker", default=YAHOO_TICKER, help="Yahoo Finance ticker.")
    parser.add_argument("--start", default=START_DATE, help="Start date, YYYY-MM-DD.")
    parser.add_argument(
        "--end",
        default="2025-01-01",
        help="Exclusive end date, YYYY-MM-DD. Defaults to 2025-01-01 to cover the repo's 2015-2024 window.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(RAW_DATA_DIR) / OUTPUT_NAME),
        help="Output CSV path.",
    )
    return parser.parse_args()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse yfinance MultiIndex columns for single-ticker downloads."""
    if isinstance(df.columns, pd.MultiIndex):
        flattened = df.copy()
        flattened.columns = flattened.columns.get_level_values(0)
        return flattened
    return df


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download and normalize Yahoo OHLCV bars into the repo schema."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(
            f"No Yahoo Finance data returned for {ticker} between {start} and {end}."
        )

    df = _flatten_columns(df)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Yahoo Finance response missing columns: {sorted(missing)}")

    output = (
        df.reset_index()[["Date", "Close", "Open", "High", "Low", "Volume"]]
        .rename(
            columns={
                "Close": "TRDPRC_1",
                "Open": "OPEN_PRC",
                "High": "HIGH_1",
                "Low": "LOW_1",
                "Volume": "ACVOL_UNS",
            }
        )
        .copy()
    )

    output["Date"] = pd.to_datetime(output["Date"]).dt.strftime("%Y-%m-%d")
    output["ACVOL_UNS"] = pd.to_numeric(output["ACVOL_UNS"], errors="coerce").fillna(0).astype("int64")
    return output[OUTPUT_COLUMNS]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prices = fetch_prices(args.ticker, args.start, args.end)
    prices.to_csv(output_path, index=False)

    print(
        f"Wrote {len(prices):,} rows to {output_path} "
        f"for {args.ticker} ({prices['Date'].min()} -> {prices['Date'].max()})"
    )


if __name__ == "__main__":
    main()
