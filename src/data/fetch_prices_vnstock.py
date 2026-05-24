"""Fetch VN-Index daily OHLCV data via vnstock quote history.

This writes the repo's expected `data/prices_VN.csv` schema:

    Date,TRDPRC_1,OPEN_PRC,HIGH_1,LOW_1,ACVOL_UNS

Usage:
    python -m src.data.fetch_prices_vnstock
    python -m src.data.fetch_prices_vnstock --start 2015-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from vnstock.api.quote import Quote

from src.config import RAW_DATA_DIR, END_DATE, START_DATE

DEFAULT_SYMBOL = "VNINDEX"
DEFAULT_SOURCE = "KBS"
OUTPUT_NAME = "prices_VN.csv"
OUTPUT_COLUMNS = ["Date", "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch VN-Index daily OHLCV data via vnstock quote history."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Index symbol for vnstock.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="vnstock quote source, e.g. KBS.")
    parser.add_argument("--start", default=START_DATE, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=END_DATE, help="End date, YYYY-MM-DD.")
    parser.add_argument(
        "--output",
        default=str(Path(RAW_DATA_DIR) / OUTPUT_NAME),
        help="Output CSV path.",
    )
    return parser.parse_args()


def fetch_prices(symbol: str, source: str, start: str, end: str) -> pd.DataFrame:
    """Download and normalize vnstock quote history into the repo schema."""
    quote = Quote(source=source, symbol=symbol)
    df = quote.history(start=start, end=end, interval="1D")
    if df is None or df.empty:
        raise ValueError(
            f"No vnstock quote data returned for {symbol} via {source} "
            f"between {start} and {end}."
        )

    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"vnstock quote history missing columns: {sorted(missing)}")

    output = (
        df[["time", "close", "open", "high", "low", "volume"]]
        .rename(
            columns={
                "time": "Date",
                "close": "TRDPRC_1",
                "open": "OPEN_PRC",
                "high": "HIGH_1",
                "low": "LOW_1",
                "volume": "ACVOL_UNS",
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

    prices = fetch_prices(args.symbol, args.source, args.start, args.end)
    prices.to_csv(output_path, index=False)

    print(
        f"Wrote {len(prices):,} rows to {output_path} "
        f"for {args.symbol} via {args.source} ({prices['Date'].min()} -> {prices['Date'].max()})"
    )


if __name__ == "__main__":
    main()
