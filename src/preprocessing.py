# bucket 1min (vwap)
# normalize prices

import numpy as np
import polars as pl

def compute_exchange_stats(src: str, lf: pl.LazyFrame) -> pl.DataFrame:
    if src == "SOLANA":
        stats_lf = (
            lf
            .group_by(["EXCHANGE", "QUOTE_ASSET"])
            .agg([
                pl.count().alias("count"),
                pl.col("block_time").min().alias("start_date"),
                pl.col("block_time").max().alias("end_date"),
            ])
            .sort(["EXCHANGE", "QUOTE_ASSET"])
        )
    
    stats_df = stats_lf.collect()
    counts = stats_df["count"].to_numpy()
    q1, q3 = np.percentile(counts, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    if lower_bound <= 0:
        lower_bound = q1

    return stats_df.filter(pl.col("count") > lower_bound)

def timestamp_to_vwap(lf: pl.LazyFrame,
    time_col: str = "block_time",
    price_col: str = "USD_PRICE",
    volume_col: str = "VOLUME",
    interval: str = "1m",
    alias: str = "vwap"
) -> pl.LazyFrame:
    """
    Take a LazyFrame of trades, truncate time to `interval`, then compute VWAP per bucket.
    Returns a LazyFrame with columns ['bucket', alias].
    """
    lf = lf.with_columns([
            pl.col(time_col).cast(pl.Datetime).dt.truncate(interval).alias("bucket")
        ])
    
    grouped = (lf
        .group_by("bucket")
        .agg([
            (pl.col(price_col) * pl.col(volume_col)).sum().alias("notional"),
            pl.col(volume_col).sum().alias("volume")
        ])
        .with_columns([
            (pl.col("notional") / pl.col("volume")).alias(alias)
        ])
        .select(["bucket", alias])
        .sort("bucket"))
    
    return grouped

def merge_vwaps(
    vwap_map: dict[str, pl.LazyFrame]
) -> pl.LazyFrame:
    """
    Given a dict of {exchange_name: LazyFrame(bucket, vwap)},
    outer-join them all on 'bucket' into one LazyFrame, with one column per exchange.
    """
    merged = None
    for name, lf in vwap_map.items():
        this = lf.rename({lf.columns[1]: name})
        if merged is None:
            merged = this
        else:
            merged = merged.join(this, on="bucket", how="outer")
            print(merged.collect_schema())
    return merged.sort("bucket")