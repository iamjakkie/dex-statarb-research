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