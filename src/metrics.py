# compute sharpe
# compute drawdown
# plots

import strategy

def compute_metrics_trades(
    trades_df: pd.DataFrame,
    initial_capital: float = 1000.0,
):
    exits = trades_df[trades_df["type"]==strategy.TradeType.EXIT].copy()

    equity = exits["capital"]
    cum_max = equity.cummax()
    dradown_pct = (cum_max - equity) / cum_max * 100.0
    strategy_return = equity.pct_change().fillna(0.0)
    ann_factor = np.sqrt(365)
    sharpe_ratio = (strategy_return.mean() / strategy_return.std()) * ann_factor

    return {
        "starting_capital": initial_capital,
        "ending_capital": equity.iloc[-1],
        "total_return": (equity.iloc[-1] - initial_capital) / initial_capital * 100.0,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": dradown_pct.max(),
        "number_of_trades": len(exits),
        "average_trade_duration": (exits["exit_time"] - exits["entry_time"]).mean().total_seconds() / 60.0,  # in minutes
        "average_trade_return": (equity.iloc[-1] - initial_capital) / len(exits) * 100.0,
        "win_rate": (exits["pnl"] > 0).mean() * 100.0,
    }