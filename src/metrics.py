# compute sharpe
# compute drawdown
# plots

import pandas as pd
import numpy as np

def compute_backtest_metrics(
    df: pd.DataFrame,
    initial_capital: float = 1000.0,
) -> dict:
    """
    Given a DataFrame with a DatetimeIndex and the columns:
      - 'position'   : {+1,0,–1} strategy position each bar
      - 'cum_strat'  : cumulative equity curve (starting at `initial_capital`)
    this function will:
      1) detect entry / exit times
      2) compute per‐trade PnL & duration
      3) compute summary metrics:
         final capital, total return %, Sharpe, max drawdown %, #trades,
         avg duration (min), avg trade return %, win rate %
    """
    # 1) detect trades
    pos = df['position']
    # entry when position goes from 0 → ±1
    entries = pos[(pos.shift(1).fillna(0) == 0) & (pos != 0)]
    # exit  when position goes from ±1 → 0
    exits   = pos[(pos.shift(1) != 0)    & (pos == 0)]

    # align the two series
    # if last bar is still in a trade, drop that open entry
    if len(entries) > len(exits):
        entries = entries.iloc[:-1]

    entry_times = entries.index
    exit_times  = exits.index[: len(entry_times)]

    trades = []
    for t0, t1 in zip(entry_times, exit_times):
        eq0 = df.at[t0, 'cum_strat']
        eq1 = df.at[t1, 'cum_strat']
        pnl = eq1 - eq0
        dur = (t1 - t0).total_seconds() / 60.0
        trades.append({
            'entry_time': t0,
            'exit_time':  t1,
            'pnl':        pnl,
            'duration_min': dur
        })
    trades_df = pd.DataFrame(trades)

    # 2) equity & drawdown curves
    equity = df['cum_strat']
    dd_pct = (equity.cummax() - equity) / equity.cummax() * 100.0

    # 3) bar‐returns for Sharpe
    bar_ret = df['cum_strat'].pct_change().fillna(0.0)
    sharpe  = bar_ret.mean() / bar_ret.std() * np.sqrt(252*24*60)

    # 4) summary stats
    ending_cap   = equity[-1]
    total_return = (ending_cap - initial_capital) / initial_capital * 100.0
    win_rate     = (trades_df['pnl'] > 0).mean() * 100.0

    return {
        'starting_capital':    initial_capital,
        'ending_capital':      ending_cap,
        'total_return_%':      total_return,
        'annualized_sharpe':   sharpe,
        'max_drawdown_%':      dd_pct.max(),
        'number_of_trades':    len(trades_df),
        'average_duration_min': trades_df['duration_min'].mean() if len(trades_df) else 0.0,
        'average_trade_return_%': trades_df['pnl'].mean() / initial_capital * 100.0,
        'win_rate_%':          win_rate,
        'trades':              trades_df  # you can inspect per‐trade stats here
    }