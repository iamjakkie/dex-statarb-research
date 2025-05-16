import pandas as pd

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
    # Work on a copy
    pdf = df.copy()

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
