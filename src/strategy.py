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
    exit_price: float
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
    col_a: str,
    col_b: str,
    z_col: str,
    entry_threshold: float,
    exit_threshold: float,
    initial_capital: float = 1000.0
) -> pd.DataFrame:
    
    capital       = initial_capital
    position      = 0
    trades        = []

    combinations = []

    if type == StrategyType.DEX_PERP:
        # get all columns where name != DEX, create sub dfs
        for col in df.columns:
            if col != "DEX":
                combinations.append((col, "DEX"))

    elif type == StrategyType.PERP_PERP:
        # permutations of all columns
        for i in range(len(df.columns)):
            for j in range(i + 1, len(df.columns)):
                col_a = df.columns[i]
                col_b = df.columns[j]
                combinations.append((col_a, col_b))

    for col_a, col_b in combinations:
        # select all rows where col_a and col_b are not null
        sub_df = features.compute_pairarb_zscore(df[[col_a, col_b, z_col]].dropna())

    for _, row in df.iterrows():
        t = row.name
        z = row[z_col]
        px_a = row[col_a]
        px_b = row[col_b]

        if position == 0:
            # if entry signal - enter position
            if z > entry_threshold:
                # only buy A asset
                entry_price = px_a
                qty = capital / entry_price
                entry_buy_px = entry_price
                capital -= qty * entry_price
                position = 1
                trade = Trade(
                    timestamp=t,
                    type=TradeType.ENTRY,
                    side="BUY A",
                    entry_price=entry_price,
                    exit_price=0.0,
                    qty=qty,
                    capital=capital
                )
                trades.append(trade)

            elif z < -entry_threshold:
                # buy B
                entry_price = px_b
                qty = capital / entry_price
                entry_buy_px = entry_price
                capital -= qty * entry_price
                position = -1
                trade = Trade(
                    timestamp=t,
                    type=TradeType.ENTRY,
                    side="BUY B",
                    entry_price=entry_price,
                    exit_price=0.0,
                    qty=qty,
                    capital=capital
                )
                trades.append(trade)

        elif position == 1:
            if z <= -exit_threshold:
                # Exit A position
                exit_price = px_b
                capital += qty * exit_price
                position = 0
                trade = Trade(
                    timestamp=t,
                    type=TradeType.EXIT,
                    side="SELL A",
                    entry_price=entry_buy_px,
                    exit_price=exit_price,
                    qty=qty,
                    capital=capital
                )
                trades.append(trade)

        elif position == -1:
            if z >= exit_threshold:
                # Exit B position
                exit_price = px_a
                capital += qty * exit_price
                position = 0
                trade = Trade(
                    timestamp=t,
                    type=TradeType.EXIT,
                    side="SELL B",
                    entry_price=entry_buy_px,
                    exit_price=exit_price,
                    qty=qty,
                    capital=capital
                )
                trades.append(trade)

    trades_df = pd.DataFrame([trade.__dict__ for trade in trades])    
    return trades_df

def pairarb_strategy(
        
)