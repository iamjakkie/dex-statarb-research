
from src import strategy
from src.backtest import Backtest


def main():
    strat = strategy.pairarb_zscore
    bt = Backtest(strat)
    bt.run()

if __name__ == "__main__":
    main()