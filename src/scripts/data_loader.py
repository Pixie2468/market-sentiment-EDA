"""
Handles ingestion and temporal alignment of the two raw data sources:
  1. Historical trade executions (millisecond-precision UNIX timestamps)
  2. Bitcoin Fear & Greed Index (daily, second-precision UNIX timestamps)

The merge strategy is specifically designed to prevent look-ahead bias:
trades are left-joined onto the most recent available sentiment reading
rather than future data.
"""

import pandas as pd

from pathlib import Path


def load_and_merge_data(trades_path: Path, fg_path: Path) -> pd.DataFrame:
    """
    Loads historical trades and sentiment index, aligns timestamps,
    and merges them without look-ahead bias.

    Parameters
    ----------
    trades_path : Path
        Filesystem path to the historical trade execution CSV.
    fg_path : Path
        Filesystem path to the Fear & Greed Index CSV.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with sentiment columns ('value', 'classification')
        joined onto each trade record by date, with forward-fill applied
        to cover any gaps in the sentiment calendar.
    """
    print("Loading datasets...")

    # Load raw CSVs into memory
    df_trades = pd.read_csv(trades_path)
    df_fg = pd.read_csv(fg_path)

    # Standardize timestamps to UTC datetime
    # Trade timestamps are in UNIX milliseconds (ms); convert to pandas datetime
    df_trades["datetime_utc"] = pd.to_datetime(df_trades["Timestamp"], unit="ms")
    # Create a date-only key (YYYY-MM-DD) for joining against daily sentiment data
    df_trades["date_key"] = df_trades["datetime_utc"].dt.strftime("%Y-%m-%d")

    # Sentiment timestamps are in UNIX seconds (s); convert to pandas datetime
    df_fg["datetime_utc"] = pd.to_datetime(df_fg["timestamp"], unit="s")
    # Same date-only key for the join
    df_fg["date_key"] = df_fg["datetime_utc"].dt.strftime("%Y-%m-%d")

    # Force chronological order prior to feature mapping and joining
    df_trades = df_trades.sort_values(by="Timestamp").reset_index(drop=True)
    df_fg = df_fg.sort_values(by="timestamp").reset_index(drop=True)

    # Merge trades with sentiment on shared date key
    # Left join: every trade keeps its row even if no sentiment data exists
    # for that date (e.g., weekends or dates outside the sentiment range).
    print("Merging and aligning timestamps...")
    df_merged = pd.merge(
        df_trades,
        df_fg[["date_key", "value", "classification"]],
        on="date_key",
        how="left",
    )

    # Forward-fill sentiment gaps
    # If a trade date has no matching sentiment entry, propagate the last
    # known value forward. This is the key mechanism that prevents look-ahead
    # bias: we never use *future* sentiment to describe *past* trades.
    df_merged["value"] = df_merged["value"].ffill()
    df_merged["classification"] = df_merged["classification"].ffill()

    # Cleanup: drop the temporary join key
    df_merged = df_merged.drop(columns=["date_key"])

    print(f"Data succesfully merged. Total records: {len(df_merged)}")
    return df_merged
