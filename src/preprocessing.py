# bucket 1min (vwap)
# normalize prices

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
    
    return stats_lf.collect()