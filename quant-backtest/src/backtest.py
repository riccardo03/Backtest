"""
Backtest engine: simulates portfolio returns with transaction costs and slippage.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    commission: float = 0.001      # 10 bps per trade (one-way)
    slippage: float = 0.0005       # 5 bps market impact (one-way)
    initial_capital: float = 1_000_000.0
    benchmark: str = "SPY"


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
) -> dict:
    """
    Simulate a strategy given daily target weights and OHLCV close prices.

    Parameters
    ----------
    weights : date × ticker, target portfolio weights (should sum ≤ 1)
    close   : date × ticker, adjusted close prices
    cfg     : cost/slippage config

    Returns
    -------
    dict with keys:
        equity      : pd.Series  — portfolio value over time
        returns     : pd.Series  — daily portfolio returns (net of costs)
        turnover    : pd.Series  — daily one-way turnover
        costs       : pd.Series  — daily transaction costs (fraction)
        weights_act : pd.DataFrame — actual weights after drift
    """
    if cfg is None:
        cfg = BacktestConfig()

    # Align on common dates and tickers
    common_dates   = weights.index.intersection(close.index)
    common_tickers = weights.columns.intersection(close.columns)
    w = weights.loc[common_dates, common_tickers].copy()
    px = close.loc[common_dates, common_tickers].copy()

    returns = px.pct_change().fillna(0.0)

    n_dates = len(common_dates)
    equity        = np.empty(n_dates)
    port_returns  = np.empty(n_dates)
    turnover_arr  = np.empty(n_dates)
    costs_arr     = np.empty(n_dates)

    equity[0]       = cfg.initial_capital
    port_returns[0] = 0.0
    turnover_arr[0] = 0.0
    costs_arr[0]    = 0.0

    # Actual holdings (weights after market drift, before rebalance)
    w_actual = w.values.copy()
    w_actual[0] = w.iloc[0].values

    for i in range(1, n_dates):
        w_prev_target = w.iloc[i - 1].values
        w_today_target = w.iloc[i].values
        r = returns.iloc[i].values

        invested = w_prev_target.sum()

        if invested < 1e-9:
            # Fully in cash: zero return, compute turnover for entry trade
            port_ret = 0.0
            w_drifted = np.zeros_like(w_prev_target)
        else:
            # Drift weights with today's returns
            w_drifted = w_prev_target * (1 + r)
            port_ret = w_drifted.sum() - invested   # P&L as fraction of NAV
            w_drifted /= w_drifted.sum()            # re-normalise for turnover calc

        # Turnover = half sum of absolute weight changes (one-way)
        to = np.abs(w_today_target - w_drifted).sum() / 2.0
        turnover_arr[i] = to

        # Transaction costs: commission + slippage on traded notional
        cost = to * (cfg.commission + cfg.slippage)
        costs_arr[i] = cost

        # Net portfolio return (relative to full NAV)
        net_ret = port_ret - cost
        port_returns[i] = net_ret
        equity[i] = equity[i - 1] * (1.0 + net_ret)

        w_actual[i] = w_today_target

    idx = common_dates
    result = {
        "equity":      pd.Series(equity, index=idx, name="equity"),
        "returns":     pd.Series(port_returns, index=idx, name="returns"),
        "turnover":    pd.Series(turnover_arr, index=idx, name="turnover"),
        "costs":       pd.Series(costs_arr, index=idx, name="costs"),
        "weights_act": pd.DataFrame(w_actual, index=idx, columns=common_tickers),
    }
    return result


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_returns(close: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
    return close[ticker].pct_change().fillna(0.0).rename("benchmark")


def benchmark_equity(close: pd.DataFrame, ticker: str = "SPY", capital: float = 1_000_000.0) -> pd.Series:
    r = benchmark_returns(close, ticker)
    return (capital * (1 + r).cumprod()).rename(f"{ticker}_equity")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    from data import load, get_close
    from features import build_feature_matrix
    from strategy import momentum_strategy, composite_strategy

    close = get_close(load())
    fm    = build_feature_matrix(close)

    cfg = BacktestConfig(commission=0.001, slippage=0.0005)

    print("Running momentum strategy...")
    w1  = momentum_strategy(fm, top_n=10, signal="mom_6m")
    r1  = run(w1, close, cfg)

    print("Running composite strategy...")
    w2  = composite_strategy(fm, top_n=10)
    r2  = run(w2, close, cfg)

    bm  = benchmark_equity(close)
    bm  = bm.reindex(r1["equity"].index)

    print("\n=== Momentum equity (last 5) ===")
    print(r1["equity"].tail())
    print("\n=== Composite equity (last 5) ===")
    print(r2["equity"].tail())
    print("\n=== Benchmark (SPY) equity (last 5) ===")
    print(bm.tail())
    print(f"\nAvg daily turnover (momentum):  {r1['turnover'].mean():.4f}")
    print(f"Avg daily costs   (momentum):  {r1['costs'].mean()*1e4:.2f} bps")
    print(f"Avg daily turnover (composite): {r2['turnover'].mean():.4f}")
    print(f"Avg daily costs   (composite): {r2['costs'].mean()*1e4:.2f} bps")
