from dataclasses import dataclass
from enum import Enum
import pandas as pd

class TradeType(Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"

@dataclass
class Trade:
    timestamp: str
    type: TradeType
    side: str
    entry_a_price: float
    entry_b_price: float
    a_qty: float
    b_qty: float
    capital: float

# zscore-based strategy
def backtest_pairarb(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    z_col: str,
    entry_z: float,
    exit_z: float,
    initial_capital: float = 1000.0
) -> pd.DataFrame:
    
    capital       = initial_capital
    position      = False
    entry_buy_px  = entry_sell_px = 0.0
    qty_buy       = qty_sell       = 0.0
    trades        = []

    for _, row in df.iterrows():
        t = row.name
        z = row[z_col]
        px_a = row[col_a]
        px_b = row[col_b]

        if not position:
            # if entry signal - enter position
            if z < entry_z:
                # buy A, sell B
                qty_buy = capital / px_a
                qty_sell = qty_buy * px_a / px_b
                entry_buy_px = px_a
                entry_sell_px = px_b
                position = True
                trades.append((t, "BUY", qty_buy, entry_buy_px))
            elif z > -entry_z:
                # buy B, sell A
                qty_sell = capital / px_b
                qty_buy = qty_sell * px_b / px_a
                entry_sell_px = px_b
                entry_buy_px = px_a
                position = True
                trades.append((t, "SELL", qty_sell, entry_sell_px))
            