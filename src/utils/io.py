"""Small dataframe I/O helpers for tabular artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_existing_table_path(path: str | Path) -> Path:
    """Return the exact table path and fail if it does not exist."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Expected table artifact at {table_path}")
    return table_path


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a dataframe from CSV or Parquet based on file suffix."""
    table_path = resolve_existing_table_path(path)
    suffix = table_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(table_path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(table_path, **kwargs)

    raise ValueError(
        f"Unsupported table format for {table_path}. Expected a .csv or .parquet file."
    )


def read_parquet_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read an exact parquet artifact and reject any other suffix."""
    table_path = resolve_existing_table_path(path)
    if table_path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a parquet artifact, got {table_path}")
    return pd.read_parquet(table_path, **kwargs)
