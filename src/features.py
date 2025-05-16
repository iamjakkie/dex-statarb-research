# src/features.py

import pandas as pd

def compute_pairarb_zscore(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    window: int = 60,
    k: float = 2.0,
    zscore_col: str = "zscore"
) -> pd.DataFrame:
    """
    Compute rolling z-score from the ratio of two price series.

    Args:
      df         : minute-indexed DataFrame containing dex_col and perp_col
      dex_col    : column name of the DEX price series
      perp_col   : column name of the perpetual price series
      window     : rolling window size for mean and std
      k          : multiplier (unused, kept for signature)
      zscore_col : name for the output z-score column

    Returns:
      DataFrame with the **same columns** as the input, plus a new column `zscore_col`.
      Drops any intermediate columns (`ratio`, `ratio_ma`, `ratio_std`).
    """
    ratio = df[col_a] / df[col_b]
    mean  = ratio.rolling(window, min_periods=window).mean()
    std   = ratio.rolling(window, min_periods=window).std()

    df[zscore_col] = (ratio - mean) / std