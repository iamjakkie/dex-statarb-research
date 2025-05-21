from dataclasses import dataclass
from enum import Enum
import pandas as pd

import features

class TradeType(Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"

@dataclass
class Trade:
    timestamp: str
    type: TradeType
    side: str
    entry_price: float
    exit_price: float | None
    qty: float
    capital: float

class StrategyType(Enum):
    DEX_PERP = "DEX_PERP"
    PERP_PERP = "PERP_PERP"

class StrategySides(Enum):
    LONGSHORT = "LONG_SHORT"
    LONG = "LONG"
    SHORT = "SHORT"

# zscore-based strategy
def backtest_pairarb(
    df: pd.DataFrame,
    type: StrategyType,
    entry_threshold: float = 2.5,
    exit_threshold: float = 1.5,
    initial_capital: float = 1000.0
) -> dict[str, pd.DataFrame]:
    
    combinations = []
    trades = {}

    if type == StrategyType.DEX_PERP:
        # get all columns where name != DEX, create sub dfs
        for col in df.columns:
            if col != "DEX":
                combinations.append(("DEX", col))

    elif type == StrategyType.PERP_PERP:
        # permutations of all columns
        for i in range(len(df.columns)):
            for j in range(i + 1, len(df.columns)):
                col_a = df.columns[i]
                col_b = df.columns[j]
                combinations.append((col_a, col_b))

    for col_a, col_b in combinations:
        print(f"Backtesting {col_a} vs {col_b}...")
        # select all rows where col_a and col_b are not null
        sub_df = features.compute_pairarb_zscore(df[[col_a, col_b]].dropna(), col_a, col_b)
        trades[col_a + "_" + col_b] = pairarb_strategy(
            sub_df,
            col_a,
            col_b,
            entry_threshold,
            exit_threshold,
            initial_capital
        )

    return trades
    

def pairarb_strategy(
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
                # always buy/long A asset
                # DEX is never a B asset, so it always has to be perp
                # and can be shorted
                entry_price_a = px_a
                a_qty = (capital/2) / entry_price_a
                capital -= a_qty * entry_price_a
                position = 1 # this flag just indicates if this is LONG A SHORT B or LONG B SHORT A
                trade = Trade(
                    timestamp=t,
                    type=TradeType.ENTRY,
                    side="LONG A",
                    entry_price=entry_price_a,
                    exit_price=None,
                    qty=a_qty,
                    capital=capital
                )
                trades.append(trade)
                entry_price_b = px_b
                b_qty = capital / entry_price_b
                capital -= b_qty * entry_price_b
                trade = Trade(
                    timestamp=t,
                    type=TradeType.ENTRY,
                    side="SHORT B",
                    entry_price=entry_price_b,
                    exit_price=None,
                    qty=b_qty,
                    capital=capital
                )
                trades.append(trade)

            elif z < -entry_threshold:
                # always buy/long B asset
                # Check if A asset is a perp or DEX
                # if DEX, then we can't short it
                # if perp, then we can short it
                entry_price_b = px_b
                b_qty = (capital/2) / entry_price_b
                capital -= b_qty * entry_price_b
                position = -1 # this flag just indicates if this is LONG A SHORT B or LONG B SHORT A
                trade = Trade(
                    timestamp=t,
                    type=TradeType.ENTRY,
                    side="LONG B",
                    entry_price=entry_price_b,
                    exit_price=None,
                    qty=b_qty,
                    capital=capital
                )
                trades.append(trade)
                if col_a != "DEX":
                    entry_price_a = px_a
                    a_qty = capital / entry_price_a
                    capital -= a_qty * entry_price_a
                    trade = Trade(
                        timestamp=t,
                        type=TradeType.ENTRY,
                        side="SHORT A",
                        entry_price=entry_price_a,
                        exit_price=None,
                        qty=a_qty,
                        capital=capital
                    )
                    trades.append(trade)

        elif position == 1:
            # if exit signal - exit position
            if z <= -exit_threshold:
                # Exit LONG A position
                # Exit SHORT B position
                exit_price = px_a
                capital += a_qty * exit_price
                position = 0
                trade = Trade(
                    timestamp=t,
                    type=TradeType.EXIT,
                    side="SELL A",
                    entry_price=entry_price_a,
                    exit_price=exit_price,
                    qty=a_qty,
                    capital=capital
                )
                trades.append(trade)
                exit_price = px_b
                capital += b_qty * exit_price
                trade = Trade(
                    timestamp=t,
                    type=TradeType.EXIT,
                    side="BUY B",
                    entry_price=entry_price_b,
                    exit_price=exit_price,
                    qty=b_qty,
                    capital=capital
                )
                trades.append(trade)

        elif position == -1:
            # if exit signal - exit position
            if z >= exit_threshold:
                # Exit LONG B position
                # Exit SHORT A position
                exit_price = px_b
                capital += b_qty * exit_price
                position = 0
                trade = Trade(
                    timestamp=t,
                    type=TradeType.EXIT,
                    side="SELL B",
                    entry_price=entry_price_b,
                    exit_price=exit_price,
                    qty=b_qty,
                    capital=capital
                )
                trades.append(trade)
                if col_a != "DEX":
                    exit_price = px_a
                    capital += a_qty * exit_price
                    trade = Trade(
                        timestamp=t,
                        type=TradeType.EXIT,
                        side="BUY A",
                        entry_price=entry_price_a,
                        exit_price=exit_price,
                        qty=a_qty,
                        capital=capital
                    )
                    trades.append(trade)

    return pd.DataFrame([trade.__dict__ for trade in trades])
