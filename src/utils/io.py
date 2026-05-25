"""Small dataframe I/O helpers for tabular artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a dataframe from CSV or Parquet based on file suffix."""
    table_path = Path(path)
    suffix = table_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(table_path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(table_path, **kwargs)

    raise ValueError(
        f"Unsupported table format for {table_path}. Expected a .csv or .parquet file."
    )
