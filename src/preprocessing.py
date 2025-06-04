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
                pl.col("time").min().alias("start_date"),
                pl.col("time").max().alias("end_date"),
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

def timestamp_to_vwap(
    lf: pl.LazyFrame,
    time_col: str = "time",
    price_col: str = "price",
    volume_col: str = "volume",
    interval: str = "1m",
    alias: str = "vwap",
    funding_col: str = None,
) -> pl.LazyFrame:
    """
    Take a LazyFrame of trades, truncate time to `interval`, then compute VWAP per bucket.
    Returns a LazyFrame with columns ['bucket', alias].
    """
    lf = lf.with_columns([
           (pl.col(time_col).dt.cast_time_unit('ms')),
        ])
        
    lf = lf.with_columns([
        pl.col(time_col).dt.truncate(interval).alias("bucket")
    ])
    
    if volume_col not in lf.columns:
        lf = lf.with_columns([
            pl.lit(1).alias(volume_col)
        ])

    exprs = [
        (pl.col(price_col) * pl.col(volume_col)).sum().alias("notional"),
        pl.col(volume_col).sum().alias("volume")
    ]
    select_exprs = ["bucket", alias]

    if funding_col:
        exprs.append(
            pl.col(funding_col).sum().alias("funding")
        )
        select_exprs.append("funding")
    
    grouped = (lf
        .group_by("bucket")
        .agg(
            exprs
        )
        .with_columns([
            (pl.col("notional") / pl.col("volume")).alias(alias)
        ])
        .select(select_exprs)
        .sort("bucket"))
    
    vwap = grouped.with_columns([
        pl.col(alias)
          .fill_nan(None)                     # turn NaN → null
          .fill_null(strategy="forward")
          .cast(pl.Float64)  # convert to float32
          .alias(alias)
    ])
    return vwap

def merge_vwaps(
    vwap_map: dict[str, pl.LazyFrame]
) -> pl.LazyFrame:
    """
    Given a dict of {exchange_name: LazyFrame(bucket, vwap)},
    outer-join them all on 'bucket' into one LazyFrame, with one column per exchange.
    """
    merged = None
    for name, lf in vwap_map.items():
        if name != "DEX":
            if "funding" in lf.columns:
                this = (lf
                    .rename({lf.columns[1]: name, lf.columns[2]: f"{name}_funding"})
                    .select(["bucket", name, f"{name}_funding"])
                )
            else:
                this = (lf
                    .rename({lf.columns[1]: name})
                    .select(["bucket", name])
                )
        else:
            this = (lf
                .rename({lf.columns[1]: name})
                .select(["bucket", name])
            )
        if merged is None:
            merged = this
        else:
            merged = (merged
                      .join(this, on="bucket", how="outer")
                      .with_columns([
                          pl.coalesce("bucket", "bucket_right").alias("bucket"),
                      ])
                      .drop("bucket_right")
                      )
            
        cols_to_ffill = [c for c in merged.columns if c != "bucket"]

        merged = (
            merged
            .filter(pl.col("bucket")<pl.datetime(2025,2,28))
            .sort("bucket")
            .with_columns([
                pl.col(c).fill_null(strategy="forward").alias(c)
                for c in cols_to_ffill
            ])
        )

        # hard limit is 2025-02-28
        # merged = merged.filter(pl.col("bucket")<'2025-03-01')
    return merged.sort("bucket")