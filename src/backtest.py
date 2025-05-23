# run backtest that outputs trades_df
# size capital

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