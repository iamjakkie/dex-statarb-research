# run backtest that outputs trades_df
# size capital
import features
import metrics
import report

import pandas as pd

def run_backtest(
    strategy: callable,
    merged_df: pd.DataFrame,
    initial_capital: float = 1000.0
):
    cols = [col for col in merged_df.columns if col in ["DEX", "DYDX", "HYPERLIQUID"]]

    combinations = []
    if "DEX" in cols:
        for perp in cols[1:]:
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

        metrics_out = metrics.compute_backtest_metrics(
            strategy_df
        )

        report.generate_backtest_report_html(
            strategy_df,
            metrics_out
        )

def benchmark(trades: list):
    pnl_df = pd.DataFrame([t for t in trades if t["type"] == "exit"])
    pnl_field = "pnl" if "pnl" in pnl_df.columns else "total_pnl"
    pnl_df["cum_pnl"] = pnl_df[pnl_field].cumsum()

    # get min and max dates
    min_dt = pnl_df["time"].min()
    max_dt = pnl_df["time"].max()

    btc_benchmark = btc_pd[btc_pd['bucket'].between(min_dt, max_dt)].copy()

    # simulate btc buy and hold using {initial_amount} 
    # calculate total change in btc price and at the end of the period
    # multiply by initial amount
    first_price = btc_benchmark['avg_price'].iloc[0]
    btc_benchmark['btc_holdings'] = initial_capital / first_price
    btc_benchmark['benchmark_value'] = btc_benchmark['btc_holdings'] * btc_benchmark['avg_price']
    btc_benchmark['benchmark_value'] = btc_benchmark['benchmark_value'] - initial_capital

    # — Create figure with secondary y-axis — 
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Cumulative PnL
    fig.add_trace(
        go.Scatter(
            x=pnl_df["time"],
            y=pnl_df["cum_pnl"],
            mode="lines+markers",
            name="Cumulative PnL",
            line=dict(color="purple")
        ),
        secondary_y=False
    )

    # Trace 2: BTC avg_price benchmark
    fig.add_trace(
        go.Scatter(
            x=btc_benchmark["bucket"],
            y=btc_benchmark["benchmark_value"],
            mode="lines",
            name="BTC Avg Price",
            line=dict(color="orange")
        )
    )

    # — Layout updates — 
    fig.update_layout(
        title="Strategy Cumulative PnL vs BTC Price Benchmark",
        xaxis_title="Time",
        height=500,
        legend=dict(orientation="h", x=0.1, y=1.1)
    )

    # Y-axis titles
    fig.update_yaxes(title_text="PnL ($)",            secondary_y=False)
    fig.update_yaxes(title_text="BTC Price (USD)",   secondary_y=True)

    fig.show()