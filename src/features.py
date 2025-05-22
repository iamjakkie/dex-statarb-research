# src/features.py

import pandas as pd

def compute_pairarb_zscore(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    funding_col: str = None,
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

    df['spread'] = df[col_a] - df[col_b]

    # 2) EWMA‐based MACD on the spread
    df['ewma_fast']   = df['spread'].ewm(span=20, adjust=False).mean()
    df['ewma_slow']   = df['spread'].ewm(span=100, adjust=False).mean()
    df['macd']        = df['ewma_fast'] - df['ewma_slow']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 3) Bollinger Bands on spread
    w = 60
    df['spread_ma']  = df['spread'].rolling(w).mean()
    df['spread_std'] = df['spread'].rolling(w).std()
    df['bb_upper']   = df['spread_ma'] + 2*df['spread_std']
    df['bb_lower']   = df['spread_ma'] - 2*df['spread_std']

    # 4) RSI on the spread
    delta = df['spread'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    # Wilder’s smoothing:
    roll = 60
    avg_up   = up.ewm(alpha=1/roll, adjust=False).mean()
    avg_down = down.ewm(alpha=1/roll, adjust=False).mean()
    rs        = avg_up/avg_down
    df['rsi'] = 100 - (100/(1+rs))

    if funding_col:

        # 5) Funding‐rate trend
        df['fund_ema_fast'] = df[funding_col].ewm(span=30, adjust=False).mean()
        df['fund_ema_slow'] = df[funding_col].ewm(span=100, adjust=False).mean()
        df['fund_trend']    = df['fund_ema_fast'] - df['fund_ema_slow']

        # funding rolling stats
        df['fund_ma']  = df[funding_col].rolling(w).mean()
        df['fund_std'] = df[funding_col].rolling(w).std()

        # funding z-score
        df['fund_z']   = (df[funding_col] - df['fund_ma']) / df['fund_std']

    return df
    