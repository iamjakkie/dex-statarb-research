# run backtest that outputs trades_df
# size capital
import features
import metrics
import report
from src import preprocessing, utils
import tokens

import pandas as pd
import polars as pl


class Backtest:
    def __init__(self, strategy, initial_capital: float = 1000.0):
        self.tokens = tokens.TOKEN_MAPPING
        self.strategy = strategy
        self.initial_capital = initial_capital

    def _run_backtest(
        self,
        token: str,
        merged_df: pd.DataFrame,
    ):
        cols = sorted(
            [col for col in merged_df.columns if col in ["DEX", "DYDX", "HYPERLIQUID"]]
        )
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
                funding_col=(
                    f"{combination[1]}_funding"
                    if f"{combination[1]}_funding" in merged_df.columns
                    else None
                ),
            )

            strategy_df = self.strategy(
                features_df,
                combination[0],
                combination[1],
                notional=self.initial_capital
            )

            if strategy_df.empty:
                continue

            benchmark = self._get_benchmark(strategy_df)

            metrics_out = metrics.compute_backtest_metrics(strategy_df)

            output_path = (
                f"{token}_{combination[0]}_{combination[1]}_backtest_report.html"
            )

            report.generate_backtest_report_html(
                token, combination, strategy_df, metrics_out, benchmark, output_path
            )

    def _get_benchmark(
        self,
        strategy_df: pd.DataFrame,
    ) -> pd.Series:
        SEC_MAX = 1e12
        MS_MAX = 1e14
        min_dt = strategy_df.index.min()
        max_dt = strategy_df.index.max()

        # define glob path between min and max dates

        btc = pl.scan_parquet("s3://iamjakkie-public/prices/BTC/*.parquet")
        btc = (
            btc.with_columns(
                [
                    pl.when(pl.col("close_time") < SEC_MAX)
                    .then(pl.col("close_time") * 1_000)  # seconds → ms
                    .when(pl.col("close_time") > MS_MAX)
                    .then(pl.col("close_time") / 1_000)  # μs → ms
                    .otherwise(pl.col("close_time"))  # already ms
                    .cast(pl.Datetime("ms"))  # interpret as ms
                    .alias("datetime_ms")
                ]
            )
            .with_columns([pl.col("datetime_ms").dt.truncate("1m").alias("bucket")])
            .filter(pl.col("bucket").is_between(min_dt, max_dt))
            .group_by("bucket")
            .agg(
                [
                    pl.col("close").mean().alias("avg_price"),
                ]
            )
            .sort("bucket")
        )
        btc_benchmark = btc.collect().to_pandas()
        # simulate btc buy and hold using {initial_amount}
        first_price = btc_benchmark["avg_price"].iloc[0]
        btc_benchmark["btc_holdings"] = self.initial_capital / first_price
        btc_benchmark["benchmark_value"] = (
            btc_benchmark["btc_holdings"] * btc_benchmark["avg_price"]
        )
        btc_benchmark["benchmark_value"] = btc_benchmark["benchmark_value"]
        btc_benchmark.set_index("bucket", inplace=True)

        return btc_benchmark["benchmark_value"]

    def run(self):
        for token, data in self.tokens.items():
            if not data["sol"]:
                continue

            dex = utils.load_data("SOLANA", token)
            dydx = utils.load_data("DYDX", token)
            hl = utils.load_data("HYPERLIQUID", token)

            dex_filtered = preprocessing.select_dex("SOLANA", dex)

            dex_vwap = preprocessing.timestamp_to_vwap(dex_filtered)
            dydx_vwap = preprocessing.timestamp_to_vwap(dydx)
            hl_vwap = preprocessing.timestamp_to_vwap(hl)

            dfs = {
                "DYDX": dydx_vwap,
                "HYPERLIQUID": hl_vwap,
                "DEX": dex_vwap
            }

            merged = preprocessing.merge_vwaps(dfs)

            self._run_backtest(
                token,
                merged
            )


