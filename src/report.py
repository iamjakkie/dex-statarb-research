import io, base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_backtest_report_html(
    combination: tuple[str, str],
    df: pd.DataFrame,
    metrics: dict,
    benchmark_series: pd.Series = None,
    output_path: str = "backtest_report.html",
):
    """
    Build a 1–2 page HTML report with:
      • Summary metrics
      • Equity curve (vs. optional benchmark)
      • Drawdown %
      • Trade PnL distribution

    Args:
      df              : your minute‐indexed DataFrame, must include 'cum_strat'
      metrics         : output of compute_backtest_metrics(df)
      benchmark_series: optional pd.Series of cumulative benchmark equity (same index)
      output_path     : where to write the HTML file
    """
    ex_a, ex_b = combination
    print(f"▶ Generating report for {ex_a} vs {ex_b}")
    imgs = {}

    # Price comparison
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df[ex_a], label=f"{ex_a} Price", color="tab:blue", alpha=0.8)
    ax.plot(df.index, df[ex_b], label=f"{ex_b} Price", color="tab:orange", alpha=0.8)
    ax.set_title(f"{ex_a} Price vs. {ex_b} Price")
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    imgs['price_comparison'] = base64.b64encode(buf.getvalue()).decode('ascii')

    # Ratio
    ratio = df[ex_a] / df[ex_b]
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(df.index, ratio, label=f"{ex_a}/{ex_b}", color="tab:purple", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"Price Ratio: {ex_a} ÷ {ex_b}")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    imgs['price_ratio'] = base64.b64encode(buf.getvalue()).decode('ascii')

    # Equity Curve
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['cum_strat'], label="Strategy", lw=2)
    if benchmark_series is not None:
        ax.plot(benchmark_series.index, benchmark_series.values,
                "--", label="BTC Hold", color="gray", alpha=0.7)
    ax.set_title("Equity Curve")
    ax.legend(loc="upper left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    imgs['equity'] = base64.b64encode(buf.getvalue()).decode('ascii')

    # Drawdown %
    dd = (df['cum_strat'].cummax() - df['cum_strat']) / df['cum_strat'].cummax() * 100
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(df.index, dd, color="red")
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("%")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    imgs['drawdown'] = base64.b64encode(buf.getvalue()).decode('ascii')

    # Trade PnL histogram
    trades_df = metrics['trades']
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(trades_df['pnl'], bins=20, edgecolor='black')
    ax.set_title("Per-Trade PnL")
    ax.set_xlabel("PnL ($)"); ax.set_ylabel("Count")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    imgs['hist'] = base64.b64encode(buf.getvalue()).decode('ascii')

    # Position timeline
    fig, ax = plt.subplots(figsize=(10,3))
    # Draw a step plot of the position (+1, 0, or –1) over time
    ax.step(df.index, df['position'], where="post", color="tab:purple")
    ax.set_ylim(-1.2, +1.2)
    ax.set_yticks([-1, 0, +1])
    ax.set_yticklabels(["Long A/Short B", "Flat", "Long B/Short A"])
    ax.set_title("Position Over Time")
    ax.set_ylabel("Position")
    ax.set_xlabel("Time")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    imgs['position_timeline'] = base64.b64encode(buf.getvalue()).decode('ascii')

    # "Spread"
    spread = df['DEX'] - df['DYDX']

    # Find the exact timestamps of entries/exits (just like in compute_metrics_trades):
    pos = df['position']
    entry_mask = (pos != 0) & (pos.shift(1).fillna(0) == 0)
    exit_mask  = (pos == 0) & (pos.shift(1).fillna(0) != 0)
    entry_times = df.index[entry_mask]
    exit_times  = df.index[exit_mask]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, spread, color='tab:gray', label="Spread (DEX − DYDX)")
    ax.set_ylabel("Spread")
    ax.set_xlabel("Time")
    ax.set_title("Spread with Entry (▲) / Exit (▼) Points")

    # Draw entry markers (green ▲)
    ax.scatter(entry_times, spread.loc[entry_times], marker="^", color="green", s=80, label="Entry")

    # Draw exit markers (red ▼)
    ax.scatter(exit_times,  spread.loc[exit_times],  marker="v", color="red",   s=80, label="Exit")

    ax.legend(loc="upper left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    imgs['spread_with_signals'] = base64.b64encode(buf.getvalue()).decode('ascii')
    
    # Rolling Volatility (60‐minute window)
    window = 60

    # 60‐minute rolling vol of your strategy returns (annualized)
    rolling_vol_strat = df['strat_ret'].rolling(window).std() * np.sqrt(252 * 24 * 60)

    # 60‐minute rolling vol of perp‐hold returns
    r_perp = df['DYDX'].pct_change().fillna(0)
    rolling_vol_perp = r_perp.rolling(window).std() * np.sqrt(252 * 24 * 60)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, rolling_vol_strat, label="Rolling Vol (Strategy)", color="tab:blue")
    ax.plot(df.index, rolling_vol_perp, label="Rolling Vol (Hold Perp)", color="tab:orange", alpha=0.7)
    ax.set_ylabel("Annualized Volatility")
    ax.set_xlabel("Time")
    ax.set_title("60‐minute Rolling Volatility: Strat vs. Hold Perp")
    ax.legend(loc="upper left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    imgs['rolling_vol'] = base64.b64encode(buf.getvalue()).decode('ascii')

    # Write out HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Backtest Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 40px; }}
    h1 {{ border-bottom: 1px solid #ccc; }}
    .section {{ margin-top: 30px; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Backtest Report</h1>

  <div class="section">
    <h2>Summary Metrics</h2>
    <ul>
      <li>Starting Capital: ${metrics['starting_capital']:.0f}</li>
      <li>Ending Capital:   ${metrics['ending_capital']:.0f}</li>
      <li>Total Return:     {metrics['total_return_%']:.2f}%</li>
      <li>Sharpe Ratio:     {metrics['annualized_sharpe']:.2f}</li>
      <li>Max Drawdown:     {metrics['max_drawdown_%']:.2f}%</li>
      <li># Trades:         {metrics['number_of_trades']}</li>
      <li>Win Rate:         {metrics['win_rate_%']:.2f}%</li>
      <li>Avg Duration:     {metrics['average_duration_min']:.1f} min</li>
      <li>Avg Trade Ret:    {metrics['average_trade_return_%']:.2f}%</li>
    </ul>
  </div>

  <div class="section">
    <h2>Price Comparison</h2>
    <img src="data:image/png;base64,{imgs['price_comparison']}" />
  </div>

  <div class="section">
    <h2>Price Ratio: {ex_a} ÷ {ex_b}</h2>
    <img src="data:image/png;base64,{imgs['price_ratio']}" />
  </div>

  <div class="section">
    <h2>Equity Curve</h2>
    <img src="data:image/png;base64,{imgs['equity']}" />
  </div>

  <div class="section">
    <h2>Drawdown (%)</h2>
    <img src="data:image/png;base64,{imgs['drawdown']}" />
  </div>

  <div class="section">
    <h2>Per‐Trade PnL Distribution</h2>
    <img src="data:image/png;base64,{imgs['hist']}" />
  </div>

  <div class="section">
    <h2>Position Over Time</h2>
    <img src="data:image/png;base64,{imgs['position_timeline']}" />
  </div>

  <div class="section">
    <h2>Spread with Entry/Exit Points</h2>
    <img src="data:image/png;base64,{imgs['spread_with_signals']}" />
  </div>

  <div class="section">
    <h2>Rolling Volatility (60‐minute window)</h2>
    <img src="data:image/png;base64,{imgs['rolling_vol']}" />
  </div>


</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)

    print(f"▶ Report written to {output_path}")