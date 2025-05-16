import pandas as pd
import numpy as np

def winsorize_and_smooth(
    df: pd.DataFrame,
    span: int = 5,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    freq: str = "1D"
) -> pd.DataFrame:
    """
    Take a DataFrame indexed by datetime and numeric columns.
    1) Winsorize each column per `freq` period to the [`lower_quantile`, `upper_quantile`] range.
    2) Apply EWMA smoothing with given `span`.

    Returns a new DataFrame of the same shape, with spikes removed and data smoothed.
    """
    pdf = df.copy()
    pdf.set_index("bucket", inplace=True)
    # Compute rolling quantiles for each period
    lower = pdf.resample(freq).quantile(lower_quantile)
    upper = pdf.resample(freq).quantile(upper_quantile)

    # Winsorize per column
    for col in pdf.columns:
        pdf[col] = (
            pdf[col]
            .combine(lower[col], lambda x, l: max(x, l))
            .combine(upper[col], lambda x, h: min(x, h))
        )

    # EWMA smoothing
    smoothed = pdf.ewm(span=span, adjust=False).mean()
    return smoothed

def despike_pct_change(
    df: pd.DataFrame,
    pct_threshold: float = 0.25,
    ignore_cols: list[str] = None
) -> pd.DataFrame:
    """
    Remove spikes by capping any numeric column’s pct‐change > pct_threshold
    relative to the previous row.  Non‐numeric (e.g. datetime) columns are left untouched.
    """
    pdf = df.copy()
    ignore = set(ignore_cols or [])
    num_cols = pdf.select_dtypes(include="number").columns.difference(ignore)

    for col in num_cols:
        # compute percent change
        pc = pdf[col].pct_change().fillna(0)
        mask = pc.abs() > pct_threshold
        if mask.any():
            # null out those outliers and ffill
            pdf.loc[mask, col] = np.nan
            pdf[col] = pdf[col].ffill()
    return pdf


def clip_outliers_iqr(
    df: pd.DataFrame,
    window: str | int = "5T",
    k: float = 3.0
) -> pd.DataFrame:
    """
    Clip any point lying outside ±k·IQR of the rolling median over `window`.
    `window` can be:
      • an integer (number of rows),
      • or a pandas offset string like '5T' (5 minutes) *if* your index is 1 min freq.
    """
    pdf = df.copy()
    pdf.set_index("bucket", inplace=True)
    
    # For each numeric column, apply rolling IQR clipping
    num_cols = pdf.select_dtypes(include='number').columns
    for col in num_cols:
        med = pdf[col].rolling(window, min_periods=1).median()
        q1  = pdf[col].rolling(window, min_periods=1).quantile(0.25)
        q3  = pdf[col].rolling(window, min_periods=1).quantile(0.75)
        iqr = q3 - q1

        lower = med - k * iqr
        upper = med + k * iqr
        pdf[col] = pdf[col].clip(lower, upper)

    return pdf

def clip_by_row_median(df, max_dev=0.3):
    """
    For each timestamp (row), compute the cross-column median.
    Then any cell where value > median*(1+max_dev) or < median*(1-max_dev)
    gets set back to the median.
    """
    pdf = df.copy()
    # compute row medians
    med = pdf.median(axis=1)
    # compute upper / lower bounds
    upper = med * (1 + max_dev)
    lower = med * (1 - max_dev)

    # for each column, clip to [lower, upper]
    for col in pdf.columns:
        pdf[col] = np.minimum(pdf[col], upper)
        pdf[col] = np.maximum(pdf[col], lower)

    return pdf

def interpolate_spikes(
    df: pd.DataFrame,
    pct_threshold: float = 0.5,
    cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Find minute-bars where price jumps by more than pct_threshold (e.g. 50%),
    mask them (and their immediate successors) as NaN, then linearly interpolate
    to fill those gaps.

    Args:
      df             : minute-indexed DataFrame with numeric price columns
      pct_threshold  : absolute fractional jump to treat as a spike (0.5 = 50%)
      cols           : list of columns to process; defaults to all numeric
    Returns:
      DataFrame with same index/columns, but spikes replaced by interpolated values.
    """
    pdf = df.copy()
    pdf.set_index("bucket", inplace=True)
    # choose columns
    num = pdf.select_dtypes(include="number").columns.tolist()
    to_proc = cols or num

    for col in to_proc:
        s = pdf[col]

        # 1) compute abs pct change vs prior bar
        jump = s.pct_change().abs().fillna(0)

        # 2) mark spikes: where jump > threshold
        mask = jump > pct_threshold

        # also mask the *bar itself* and optionally the *one after* so we catch
        # prolonged identical bad values
        bad = mask.copy()
        bad |= mask.shift(-1, fill_value=False)

        # 3) mask them as NaN
        s_clean = s.mask(bad)

        # 4) linear interpolate (by time index)
        pdf[col] = s_clean.interpolate(method="time")

    return pdf