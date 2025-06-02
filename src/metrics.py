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
    
    # Mask for entries: pos_t != 0 but pos_{t−1} == 0
    entry_mask = (pos != 0) & (pos.shift(1).fillna(0) == 0)
    # Mask for exits: pos_t == 0 but pos_{t−1} != 0
    exit_mask  = (pos == 0) & (pos.shift(1).fillna(0) != 0)
    
    entry_times = df.index[entry_mask]
    exit_times  = df.index[exit_mask]
    
    # If the last entry never got closed, drop that “open” entry.
    if len(entry_times) > len(exit_times):
        entry_times = entry_times[:-1]
    
    # Now pair them one-to-one, ensuring that exit always follows entry
    trades_list = []
    for t0, t1 in zip(entry_times, exit_times):
        # sanity check: if an exit somehow came before entry, skip it
        if t1 <= t0:
            continue
        
        # Equity at entry vs equity at exit
        eq0 = df.at[t0, 'cum_strat']
        eq1 = df.at[t1, 'cum_strat']
        pnl = eq1 - eq0
        
        # duration in minutes
        dur_min = (t1 - t0).total_seconds() / 60.0
        
        trades_list.append({
            'entry_time':    t0,
            'exit_time':     t1,
            'pnl':           pnl,
            'duration_min':  dur_min
        })
    
    trades_df = pd.DataFrame(trades_list)
    
    # --- 2) Compute drawdown curve (in %)
    equity = df['cum_strat']  # e.g. starts at 1000
    # rolling max of equity
    cummax_equity = equity.cummax()
    # drawdown in dollars
    dd_dollars = cummax_equity - equity
    # drawdown in percent
    dd_percent = (dd_dollars / cummax_equity) * 100.0
    
    # --- 3) Compute Sharpe using the actual strategy bar returns
    #     We assume strat_ret is a fractional PnL (e.g. 0.001 = +0.1% in that minute).
    bar_ret = df['strat_ret'].fillna(0.0)
    # annualization factor: sqrt(252 trading days * 24 hours/day * 60 min/hour)
    ann_factor = np.sqrt(252 * 24 * 60)
    # if stdev is zero, Sharpe = nan
    if bar_ret.std(ddof=0) > 0:
        sharpe = (bar_ret.mean() / bar_ret.std(ddof=0)) * ann_factor
    else:
        sharpe = np.nan
    
    # --- 4) Compute all remaining summary stats
    ending_cap    = equity.iloc[-1]
    total_return  = (ending_cap - initial_capital) / initial_capital * 100.0
    
    num_trades    = len(trades_df)
    if num_trades > 0:
        avg_dur    = trades_df['duration_min'].mean()
        # average trade return as percent of initial capital
        avg_trade_return = (trades_df['pnl'].mean() / initial_capital) * 100.0
        win_rate   = (trades_df['pnl'] > 0).mean() * 100.0
    else:
        avg_dur = 0.0
        avg_trade_return = 0.0
        win_rate = 0.0
    
    max_dd_pct    = dd_percent.max()
    
    return {
        'starting_capital':     initial_capital,
        'ending_capital':       ending_cap,
        'total_return_%':       total_return,
        'annualized_sharpe':    sharpe,
        'max_drawdown_%':       max_dd_pct,
        'number_of_trades':     num_trades,
        'average_duration_min': avg_dur,
        'average_trade_return_%': avg_trade_return,
        'win_rate_%':           win_rate,
        'trades':            trades_df  # per‐trade details
    }