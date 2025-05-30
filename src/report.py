import io, base64
import matplotlib.pyplot as plt
import pandas as pd

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

    # 1) Equity Curve
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

    # 2) Drawdown %
    dd = (df['cum_strat'].cummax() - df['cum_strat']) / df['cum_strat'].cummax() * 100
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(df.index, dd, color="red")
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("%")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    imgs['drawdown'] = base64.b64encode(buf.getvalue()).decode('ascii')

    # 3) Trade PnL histogram
    trades_df = metrics['trades']
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(trades_df['pnl'], bins=20, edgecolor='black')
    ax.set_title("Per-Trade PnL")
    ax.set_xlabel("PnL ($)"); ax.set_ylabel("Count")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    imgs['hist'] = base64.b64encode(buf.getvalue()).decode('ascii')

    # 4) Write out HTML
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

</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)

    print(f"▶ Report written to {output_path}")