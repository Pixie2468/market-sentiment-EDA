"""
Engineers baseline financial features on top of the raw merged dataset.
These features form the analytical foundation for all downstream analyses:

  - Net_PnL:         True economic profit after subtracting exchange fees
  - Execution_Type:  Human-readable label ('Taker' or 'Maker') derived
                     from the boolean 'Crossed' column
"""

import numpy as np
import pandas as pd


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the baseline financial features required for Phase 1 analysis:
    Net PnL and Execution Type.

    Parameters
    ----------
    df : pd.DataFrame
        The merged trades + sentiment DataFrame produced by
        ``load_and_merge_data()``. Must contain columns
        'Closed PnL', 'Fee', and 'Crossed'.

    Returns
    -------
    pd.DataFrame
        A copy of the input with two new columns appended:
        - 'Net_PnL':         Closed PnL minus the exchange fee
        - 'Execution_Type':  'Taker' if Crossed is True, else 'Maker'
    """
    # Work on a copy to avoid mutating the caller's DataFrame
    df = df.copy()

    # Net PnL: friction-adjusted profitability
    # Raw 'Closed PnL' ignores trading friction (fees). Subtracting the
    # exchange 'Fee' gives the true economic result of each trade.
    df["Net_PnL"] = df["Closed PnL"] - df["Fee"]

    # Execution Type: Taker vs Maker label
    # 'Crossed' == True means the order crossed the bid-ask spread (a "Taker"
    # or aggressive/market order that pays a higher fee for instant execution).
    # 'Crossed' == False means the order rested on the book (a "Maker" or
    # passive/limit order that provides liquidity).
    df["Execution_Type"] = np.where(df["Crossed"] == True, "Taker", "Maker")

    return df
