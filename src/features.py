# src/features.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

def compute_pairarb_features(
    merged_df: pd.DataFrame,
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
    df = merged_df.copy()
    # 1) z‐score of price ratio + rolling std
    ratio = df[col_a] / df[col_b]
    mu    = ratio.rolling(window, min_periods=window).mean()
    sigma = ratio.rolling(window, min_periods=window).std()
    df[zscore_col]          = (ratio - mu) / sigma
    df[f"{zscore_col}_std"] = sigma

    # 2) spread & volatility
    df["spread"] = df[col_a] - df[col_b]
    df["vol"]    = df["spread"].rolling(window, min_periods=window).std()

    # 3) EWMA‐based MACD on spread
    df["ewma_fast"]   = df["spread"].ewm(span=20, adjust=False).mean()
    df["ewma_slow"]   = df["spread"].ewm(span=100, adjust=False).mean()
    df["macd"]        = df["ewma_fast"] - df["ewma_slow"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 4) Bollinger Bands on spread
    df["spread_ma"]  = df["spread"].rolling(window, min_periods=window).mean()
    df["spread_std"] = df["spread"].rolling(window, min_periods=window).std()
    df["bb_upper"]   = df["spread_ma"] + 2 * df["spread_std"]
    df["bb_lower"]   = df["spread_ma"] - 2 * df["spread_std"]

    # 5) RSI on the spread
    delta = df["spread"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll = window
    avg_up   = up.ewm(alpha=1/roll, adjust=False).mean()
    avg_down = down.ewm(alpha=1/roll, adjust=False).mean()
    rs       = avg_up / avg_down
    df["rsi"] = 100 - (100 / (1 + rs))

    # 6) rolling “OLS” hedge ratio & residual (manual)
    logA = np.log(df[col_a])
    logB = np.log(df[col_b])
    cov  = logA.rolling(window, min_periods=window).cov(logB)
    var  = logB.rolling(window, min_periods=window).var()
    df["beta_ols"]  = cov / var
    df["resid_ols"] = logA - df["beta_ols"] * logB

    # 7) optional funding features
    if funding_col and funding_col in df:
        df["fund_ema_fast"] = df[funding_col].ewm(span=30, adjust=False).mean()
        df["fund_ema_slow"] = df[funding_col].ewm(span=100, adjust=False).mean()
        df["fund_trend"]    = df["fund_ema_fast"] - df["fund_ema_slow"]

        fund_mu  = df[funding_col].rolling(window, min_periods=window).mean()
        fund_std = df[funding_col].rolling(window, min_periods=window).std()
        df["fund_ma"]  = fund_mu
        df["fund_std"] = fund_std
        df["fund_z"]   = (df[funding_col] - fund_mu) / fund_std

    # 8) drop rows with any NaNs in the core features
    required = [
        zscore_col, f"{zscore_col}_std",
        "spread", "vol",
        "ewma_fast", "ewma_slow", "macd", "macd_signal",
        "spread_ma", "spread_std", "bb_upper", "bb_lower",
        "rsi", "beta_ols", "resid_ols"
    ]
    if funding_col and funding_col in df:
        required += ["fund_trend", "fund_ma", "fund_std", "fund_z"]

    df.set_index("bucket", inplace=True)

    return df.dropna(subset=required)
    