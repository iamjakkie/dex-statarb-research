from dataclasses import dataclass
from enum import Enum
import pandas as pd

import features

def pairarb_combined_signal(
    input_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    entry_threshold: float = 2.5,
    exit_threshold: float = 1.5,
    initial_capital: float = 1000.0
) -> pd.DataFrame:
    """
    Backtest a pair arbitrage strategy on a DataFrame of price data.
    
    Args:
        df (pd.DataFrame): DataFrame containing price data with columns ['DEX', 'PERP1', 'PERP2'].
        type (StrategyType): Type of strategy to backtest (DEX_PERP or PERP_PERP).
        entry_threshold (float): Z-score threshold for entering a trade.
        exit_threshold (float): Z-score threshold for exiting a trade.
        initial_capital (float): Initial capital for the strategy.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing trades executed during the backtest.
    """
    
    df = input_df.copy()

    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # LONG‐leg signals  (we want: z>+, macd_hist>0, fund_z>0)
    df['s_z_long'] = (df['zscore']    >  2.5).astype(int)
    df['s_m_long'] = (df['macd_hist'] >  0  ).astype(int)
    df['s_f_long'] = (df['fund_z']     >  0  ).astype(int)
    df['score_long'] = df[['s_z_long','s_m_long','s_f_long']].sum(axis=1)

    # SHORT‐leg signals (we want: z<−, macd_hist<0, fund_z<0)
    df['s_z_short'] = (df['zscore']    < -2.5).astype(int)
    df['s_m_short'] = (df['macd_hist'] <  0  ).astype(int)
    df['s_f_short'] = (df['fund_z']     <  0  ).astype(int)
    df['score_short'] = df[['s_z_short','s_m_short','s_f_short']].sum(axis=1)

    ENTRY_SCORE = 2
    EXIT_SCORE = 1

    df['enter_long']  = df['score_long']  >= ENTRY_SCORE
    df['exit_long']   = df['score_long']  <= EXIT_SCORE

    df['enter_short'] = df['score_short'] >= ENTRY_SCORE
    df['exit_short']  = df['score_short'] <= EXIT_SCORE

    min_hold = pd.Timedelta("15min")   # require at least 15 minutes per trade
    df['position']  = 0
    pos            = 0
    last_entry_ts  = pd.NaT

    for t, row in df.iterrows():
        
        # determine if we’ve held long enough
        held_long_enough = (last_entry_ts is not pd.NaT) and (t - last_entry_ts >= min_hold)

        # 1) exit logic
        if pos == +1 and row['exit_long'] and held_long_enough:
            pos = 0
        elif pos == -1 and row['exit_short'] and held_long_enough:
            pos = 0

        # 2) entry logic (only when flat)
        elif pos == 0:
            # if row['enter_long']:
            #     pos           = +1
            #     last_entry_ts = t
            if row['enter_short']:
                pos           = -1
                last_entry_ts = t

        df.at[t, 'position'] = pos

    # ── 4) bar returns
    r_dex  = df[col_a].pct_change().fillna(0)
    r_perp = df[col_b].pct_change().fillna(0)
    pos_lag = df['position'].shift(1).fillna(0).astype(int)

    strat_ret = []
    for p, rd, rp in zip(pos_lag, r_dex, r_perp):
        if p == -1:
            # realistic: long DEX + short PERP
            ret = 0.5 * (rd + (-rp))
        else:
            ret = 0.0
        strat_ret.append(ret)
    df['strat_ret'] = strat_ret

    # ── 5) equity curves
    df['cum_strat'] = (1 + df['strat_ret']).cumprod() * 1000
    df['cum_perp']  = (1 + r_perp).cumprod()    * 1000

    return df


def pairarb_zscore(
    input_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    entry_threshold: float = 2.5,
    exit_threshold: float = 1.5,
    initial_capital: float = 1000.0
) -> pd.DataFrame:
    
    df = input_df.copy()

    # LONG-leg signals
    df["score_long"] = (df["zscore"] > entry_threshold).astype(int)

    # SHORT-leg signals
    df["score_short"] = (df["zscore"] < -entry_threshold).astype(int)

    ENTRY_SCORE = 1
    EXIT_SCORE = 0

    df["enter_long"]  = df["score_long"]  >= ENTRY_SCORE
    df["exit_long"]   = df["score_long"]  <= EXIT_SCORE
    df["enter_short"] = df["score_short"] >= ENTRY_SCORE
    df["exit_short"]  = df["score_short"] <= EXIT_SCORE

    df['position']  = 0
    pos            = 0

    for t, row in df.iterrows():
        
        # 1) exit logic
        if pos == +1 and row['exit_long']:
            pos = 0
        elif pos == -1 and row['exit_short']:
            pos = 0

        # 2) entry logic (only when flat)
        elif pos == 0:
            if row['enter_short']:
                pos           = -1

        df.at[t, 'position'] = pos

    # ── 4) bar returns
    r_dex  = df[col_a].pct_change().fillna(0)
    r_perp = df[col_b].pct_change().fillna(0)
    pos_lag = df['position'].shift(1).fillna(0).astype(int)

    strat_ret = []
    for p, rd, rp in zip(pos_lag, r_dex, r_perp):
        if p == -1:
            # realistic: long DEX + short PERP
            ret = 0.5 * (rd + (-rp))
        else:
            ret = 0.0
        strat_ret.append(ret)
    df['strat_ret'] = strat_ret

    # ── 5) equity curves
    df['cum_strat'] = (1 + df['strat_ret']).cumprod() * 1000
    df['cum_perp']  = (1 + r_perp).cumprod()    * 1000

    return df

# zscore-based strategy
def backtest_pairarb(
    df: pd.DataFrame,
    entry_threshold: float = 2.5,
    exit_threshold: float = 1.5,
    initial_capital: float = 1000.0
) -> dict[str, pd.DataFrame]:
    
    combinations = []
    trades = {}

    if "DEX" in df.columns:
        # get all columns where name != DEX, create sub dfs
        for col in df.columns:
            if col not in ["DEX", "bucket"] and col.endswith("_funding") is False:
                combinations.append(("DEX", col))

    
    perps = [col for col in df.columns if col in ["DYDX", "HYPERLIQUID", "DRIFT"]]
    for i in range(len(perps)):
        for j in range(i + 1, len(perps)):
            col_a = perps[i]
            col_b = perps[j]
            combinations.append((col_a, col_b))

    print(combinations)
    for col_a, col_b in combinations:
        print(f"Backtesting {col_a} vs {col_b}...")
        # select all rows where col_a and col_b are not null
        sub_df = features.compute_pairarb_zscore(df[[col_a, col_b]].dropna(), col_a, col_b)
        trades[col_a + "_" + col_b] = pairarb_strategy_inv(
            sub_df,
            col_a,
            col_b,
            entry_threshold,
            exit_threshold,
            initial_capital
        )

    return trades
    

# def pairarb_strategy(
#     df: pd.DataFrame,
#     col_a: str,
#     col_b: str,
#     entry_threshold: float,
#     exit_threshold: float,
#     initial_capital: float
# ) -> pd.DataFrame:
#     """
#     Backtest a pair arbitrage strategy on a DataFrame of price data.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing price data with columns ['DEX', 'PERP1', 'PERP2'].
#         entry_threshold (float): Z-score threshold for entering a trade.
#         exit_threshold (float): Z-score threshold for exiting a trade.
#         initial_capital (float): Initial capital for the strategy.

#     Returns:
#         pd.DataFrame: DataFrame containing the trades executed during the backtest.
#     """
    
#     capital = initial_capital
#     position = 0
#     trades = []
#     entry_price_a = entry_price_b = None

#     for _, row in df.iterrows():
#         t = row.name
#         z = row["zscore"]
#         px_a = row[col_a]
#         px_b = row[col_b]

#         if position == 0:
#             # if entry signal enter position
#             if z > entry_threshold:
#                 # always buy/long A asset
#                 # DEX is never a B asset, so it always has to be perp
#                 # and can be shorted
#                 entry_price_a = px_a
#                 a_qty = (capital/2) / px_a
#                 capital -= a_qty * px_a
#                 entry_price_b = px_b
#                 b_qty = capital / px_b
#                 capital -= b_qty * px_b
#                 position = -1 # this flag just indicates if this is LONG A SHORT B or LONG B SHORT A
#                 trade = Trade(
#                     timestamp=t,
#                     type=TradeType.ENTRY,
#                     side="LONG B SHORT A",
#                     entry_price_a=px_a,
#                     exit_price_ae=None,
#                     entry_price_b=px_b,
#                     exit_price_b=None,
#                     qty_a=a_qty,
#                     qty_b=b_qty,
#                     capital=capital
#                 )

#             elif z < -entry_threshold:
#                 # always buy/long B asset
#                 # Check if A asset is a perp or DEX
#                 # if DEX, then we can't short it
#                 # if perp, then we can short it

#                 if col_a != "DEX":
#                     entry_price_a = px_a
#                     a_qty = (capital/2) / entry_price_a
#                     capital -= a_qty * entry_price_a
#                     entry_price_b = px_b
#                     b_qty = capital / entry_price_b
#                     capital -= b_qty * entry_price_b
#                     position = 1 # this flag just indicates if this is LONG A SHORT B or LONG B SHORT A
#                     trade = Trade(
#                         timestamp=t,
#                         type=TradeType.ENTRY,
#                         side="LONG A SHORT B",
#                         entry_price_a=entry_price_a,
#                         exit_price_a=None,
#                         entry_price_b=entry_price_b,
#                         exit_price_b=None,
#                         qty_a=a_qty,
#                         qty_b=b_qty,
#                         capital=capital
#                     )

#                 entry_price_b = px_b
#                 b_qty = (capital/2) / entry_price_b
#                 capital -= b_qty * entry_price_b
#                 position = -1 # this flag just indicates if this is LONG A SHORT B or LONG B SHORT A
#                 trade = Trade(
#                     timestamp=t,
#                     type=TradeType.ENTRY,
#                     side="LONG B",
#                     entry_price=entry_price_b,
#                     exit_price=None,
#                     qty=b_qty,
#                     capital=capital
#                 )
#                 trades.append(trade)
#                 if col_a != "DEX":
#                     entry_price_a = px_a
#                     a_qty = capital / entry_price_a
#                     capital -= a_qty * entry_price_a
#                     trade = Trade(
#                         timestamp=t,
#                         type=TradeType.ENTRY,
#                         side="SHORT A",
#                         entry_price=entry_price_a,
#                         exit_price=None,
#                         qty=a_qty,
#                         capital=capital
#                     )
#                     trades.append(trade)

#         elif position == 1:
#             # if exit signal - exit position
#             if z <= -exit_threshold and px_a > entry_price_a:
#                 # Exit LONG A position
#                 # Exit SHORT B position
#                 exit_price = px_a
#                 capital += a_qty * exit_price
#                 position = 0
#                 trade = Trade(
#                     timestamp=t,
#                     type=TradeType.EXIT,
#                     side="SELL A",
#                     entry_price=entry_price_a,
#                     exit_price=exit_price,
#                     qty=a_qty,
#                     capital=capital
#                 )
#                 trades.append(trade)
#                 exit_price = px_b
#                 capital += b_qty * exit_price
#                 trade = Trade(
#                     timestamp=t,
#                     type=TradeType.EXIT,
#                     side="BUY B",
#                     entry_price=entry_price_b,
#                     exit_price=exit_price,
#                     qty=b_qty,
#                     capital=capital
#                 )
#                 trades.append(trade)

#         elif position == -1:
#             # if exit signal - exit position
#             if z >= exit_threshold and px_b > entry_price_b:
#                 # Exit LONG B position
#                 # Exit SHORT A position
#                 exit_price = px_b
#                 capital += b_qty * exit_price
#                 position = 0
#                 trade = Trade(
#                     timestamp=t,
#                     type=TradeType.EXIT,
#                     side="SELL B",
#                     entry_price=entry_price_b,
#                     exit_price=exit_price,
#                     qty=b_qty,
#                     capital=capital
#                 )
#                 trades.append(trade)
#                 if col_a != "DEX":
#                     exit_price = px_a
#                     capital += a_qty * exit_price
#                     trade = Trade(
#                         timestamp=t,
#                         type=TradeType.EXIT,
#                         side="BUY A",
#                         entry_price=entry_price_a,
#                         exit_price=exit_price,
#                         qty=a_qty,
#                         capital=capital
#                     )
#                     trades.append(trade)

#     return pd.DataFrame([trade.__dict__ for trade in trades])

def pairarb_strategy_inv(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    entry_threshold: float,
    exit_threshold: float,
    initial_capital: float
) -> pd.DataFrame:
    """
    Backtest a pair arbitrage strategy on a DataFrame of price data.
    
    Args:
        df (pd.DataFrame): DataFrame containing price data with columns ['DEX', 'PERP1', 'PERP2'].
        entry_threshold (float): Z-score threshold for entering a trade.
        exit_threshold (float): Z-score threshold for exiting a trade.
        initial_capital (float): Initial capital for the strategy.

    Returns:
        pd.DataFrame: DataFrame containing the trades executed during the backtest.
    """
    
    capital = initial_capital
    position = 0
    trades = []
    entry_price_a = entry_price_b = None

    for _, row in df.iterrows():
        t = row.name
        z = row["zscore"]
        px_a = row[col_a]
        px_b = row[col_b]


        if position == 0:
            # if entry signal enter position
            if z > entry_threshold:
                # long B short A if possible
                # DEX is never a B asset
                # ignore this leg if col_a is DEX
                if col_a == "DEX":
                    continue
                else:
                    entry_price_a = px_a
                    a_qty = (capital/2) / px_a
                    capital -= a_qty * px_a
                    entry_price_b = px_b
                    b_qty = capital / px_b
                    capital -= b_qty * px_b
                    position = 1 # this flag just indicates if this is LONG A SHORT B or LONG B SHORT A
                    trade = Trade(
                        timestamp=t,
                        type=TradeType.ENTRY,
                        side="LONG B SHORT A",
                        entry_price_a=px_a,
                        exit_price_a=None,
                        entry_price_b=px_b,
                        exit_price_b=None,
                        qty_a=a_qty,
                        qty_b=b_qty,
                        capital=capital
                    )

            elif z < -entry_threshold:
                # long A short B if possible
                entry_price_a = px_a
                a_qty = (capital/2) / entry_price_a
                capital -= a_qty * entry_price_a
                entry_price_b = px_b
                b_qty = capital / entry_price_b
                capital -= b_qty * entry_price_b
                position = -1
                trade = Trade(
                    timestamp=t,
                    type=TradeType.ENTRY,
                    side="LONG A SHORT B",
                    entry_price_a=entry_price_a,
                    exit_price_a=None,
                    entry_price_b=entry_price_b,
                    exit_price_b=None,
                    qty_a=a_qty,
                    qty_b=b_qty,
                    capital=capital
                )
                trades.append(trade)

        elif position == 1:
            # if exit signal - exit position
            continue

        elif position == -1:
            # if exit signal - exit position
            if z >= exit_threshold or z <= -exit_threshold:
                # Exit position
                exit_price_a = px_a
                capital += a_qty * exit_price_a
                exit_price_b = px_b
                capital += b_qty * exit_price_b
                position = 0
                trade = Trade(
                    timestamp=t,
                    type=TradeType.EXIT,
                    side="CLOSE A/B",
                    entry_price_a=entry_price_a,
                    exit_price_a=exit_price_a,
                    entry_price_b=entry_price_b,
                    exit_price_b=exit_price_b,
                    qty_a=a_qty,
                    qty_b=b_qty,
                    capital=capital
                )
                trades.append(trade)

    return pd.DataFrame([trade.__dict__ for trade in trades])

def backtest_feature_pairarb(
        
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    initial_capital: float = 1_000.0,
    z_entry: float        = 2.5,
    z_exit:  float        = 1.5,
    rsi_low: float        = 30.0,
    bb_mult: float        = 2.0
) -> pd.DataFrame:
    """
    Feature-based pair arbitrage:
      - entry when zscore > z_entry
        AND macd > macd_signal
        AND rsi < rsi_low
      - exit when zscore < z_exit
        OR macd < macd_signal
        OR price spread re-enters Bollinger Bands
    """

    capital     = initial_capital
    position    = 0       # 0=no, 1=long A short B
    a_qty = b_qty = 0.0
    entry_px_a  = entry_px_b = 0.0
    trades: list[Trade] = []

    for t, row in df.iterrows():
        z      = row["zscore"]
        macd   = row["macd"]
        sig    = row["macd_signal"]
        rsi    = row["rsi"]
        spread = row["spread"]
        bb_up  = row["bb_upper"]
        bb_lo  = row["bb_lower"]
        px_a   = row[col_a]
        px_b   = row[col_b]

        # --- EXIT logic ---
        if position == 1:
            exit_cond = (
                (z < z_exit)
                or (macd < sig)
                or (spread < bb_up and spread > bb_lo)
            )
            if exit_cond:
                # unwind both legs
                pnl_a = (px_a - entry_px_a) * a_qty
                pnl_b = (entry_px_b - px_b) * b_qty
                capital += pnl_a + pnl_b
                trades.append(Trade(
                    timestamp     = t,
                    type          = TradeType.EXIT,
                    side          = "CLOSE A/B",
                    entry_price_a = entry_px_a,
                    exit_price_a  = px_a,
                    entry_price_b = entry_px_b,
                    exit_price_b  = px_b,
                    qty_a         = a_qty,
                    qty_b         = b_qty,
                    capital       = capital
                ))
                position = 0

        # --- ENTRY logic ---
        if position == 0:
            enter_cond = (
                (z > z_entry)
                and (macd > sig)
                and (rsi < rsi_low)
                and (spread > bb_up)
            )
            if enter_cond:
                # allocate half capital to each leg
                half = capital / 2
                entry_px_a = px_a
                a_qty      = half / px_a
                entry_px_b = px_b
                b_qty      = half / px_b
                capital   -= (half * 2)
                position    = 1
                trades.append(Trade(
                    timestamp     = t,
                    type          = TradeType.ENTRY,
                    side          = "LONG A / SHORT B",
                    entry_price_a = entry_px_a,
                    exit_price_a  = 0.0,
                    entry_price_b = entry_px_b,
                    exit_price_b  = 0.0,
                    qty_a         = a_qty,
                    qty_b         = b_qty,
                    capital       = capital
                ))

    return pd.DataFrame([t.__dict__ for t in trades])

