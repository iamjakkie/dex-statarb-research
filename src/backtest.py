# run backtest that outputs trades_df
# size capital
import features
import metrics
import report

import pandas as pd
import polars as pl

def run_backtest(
    token: str,
    strategy: callable,
    merged_df: pd.DataFrame,
    initial_capital: float = 1000.0
):
    cols = sorted([col for col in merged_df.columns if col in ["DEX", "DYDX", "HYPERLIQUID"]])
    print(cols)
    combinations = []
    if "DEX" in cols:
        for perp in cols[1:]:
            print(perp)
            combinations.append((cols[0], perp))
    else:
        for perp in cols:
            for perp2 in cols[1:]:
                if perp != perp2:
                    combinations.append((perp, perp2))

    for combination in combinations:
        print(f"Running backtest for {combination[0]} vs {combination[1]}")

        features_df = features.compute_pairarb_features(
            merged_df,
            col_a=combination[0],
            col_b=combination[1],
            funding_col=f"{combination[1]}_funding" if f"{combination[1]}_funding" in merged_df.columns else None
        )

        strategy_df = strategy(
            features_df,
            combination[0],
            combination[1],
        )

        if strategy_df.empty:
            continue

        benchmark = get_benchmark(strategy_df)

        metrics_out = metrics.compute_backtest_metrics(
            strategy_df
        )

        output_path = f"{token}_{combination[0]}_{combination[1]}_backtest_report.html"

        report.generate_backtest_report_html(
            combination,
            strategy_df,
            metrics_out,
            benchmark,
            output_path
        )

def get_benchmark(
        strategy_df: pd.DataFrame, 
        initial_capital: float = 1000.0
    ) -> pd.Series:
    SEC_MAX = 1e12
    MS_MAX = 1e14
    min_dt = strategy_df.index.min()
    max_dt = strategy_df.index.max()

    # define glob path between min and max dates

    btc = pl.scan_parquet("s3://iamjakkie-public/prices/BTC/*.parquet")
    btc = (
        btc
        .with_columns([
        pl.when(pl.col("close_time") < SEC_MAX)
          .then(pl.col("close_time") * 1_000)        # seconds → ms
         .when(pl.col("close_time") > MS_MAX)
          .then(pl.col("close_time") / 1_000)        # μs → ms
         .otherwise(pl.col("close_time"))            # already ms
         .cast(pl.Datetime("ms"))                    # interpret as ms
         .alias("datetime_ms")
    ])
        .with_columns([
            pl.col("datetime_ms").dt.truncate("1m").alias("bucket")
        ])
        .filter(
            pl.col("bucket").is_between(min_dt, max_dt)
        )
        .group_by("bucket")
        .agg([
            pl.col("close").mean().alias("avg_price"),
        ])
        .sort("bucket")
    )
    btc_benchmark = btc.collect().to_pandas()
    # simulate btc buy and hold using {initial_amount}
    first_price = btc_benchmark['avg_price'].iloc[0]
    btc_benchmark['btc_holdings'] = initial_capital / first_price
    btc_benchmark['benchmark_value'] = btc_benchmark['btc_holdings'] * btc_benchmark['avg_price']
    btc_benchmark['benchmark_value'] = btc_benchmark['benchmark_value']
    btc_benchmark.set_index('bucket', inplace=True)

    return btc_benchmark["benchmark_value"]