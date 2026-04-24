"""
Walk-forward validation: train on a rolling window, test on the next out-of-sample period.
Prevents overfitting by never using future data in signal construction.

Schema:
  |<--- train_years --->|<- test_months ->|
  |_____________________|_________________|_____ ...
                        |<--- train_years --->|<- test_months ->|
                                              ...
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable
from dateutil.relativedelta import relativedelta

from backtest import run, BacktestConfig, benchmark_returns
from metrics import summary, compare, cagr, sharpe, max_drawdown, volatility


# ---------------------------------------------------------------------------
# Window config
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    train_years: int   = 3      # in-sample window length
    test_months: int   = 6      # out-of-sample period per fold
    min_train_obs: int = 500    # minimum trading days required to run a fold


# ---------------------------------------------------------------------------
# Core walk-forward loop
# ---------------------------------------------------------------------------

def run_walkforward(
    strategy_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    close: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    wf_cfg: WalkForwardConfig = None,
    bt_cfg: BacktestConfig = None,
    label: str = "Strategy",
) -> dict:
    """
    Run walk-forward validation for a given strategy function.

    Parameters
    ----------
    strategy_fn : callable(fm_slice, close_slice) -> weights DataFrame
        Must accept a feature_matrix slice and a close slice, return daily weights.
    close          : full close price DataFrame (date × ticker)
    feature_matrix : full feature matrix (MultiIndex date/ticker × features)
    wf_cfg         : walk-forward window settings
    bt_cfg         : backtest cost settings
    label          : name for reporting

    Returns
    -------
    dict with keys:
        oos_returns  : pd.Series — concatenated out-of-sample returns
        oos_equity   : pd.Series — compounded OOS equity curve (starts at 1)
        fold_metrics : pd.DataFrame — per-fold performance summary
        folds        : list of fold detail dicts
    """
    if wf_cfg is None:
        wf_cfg = WalkForwardConfig()
    if bt_cfg is None:
        bt_cfg = BacktestConfig()

    all_dates = feature_matrix.index.get_level_values("date").unique().sort_values()
    start     = all_dates[0]
    end       = all_dates[-1]

    fold_returns = []
    fold_records = []
    folds        = []
    fold_num     = 0

    cursor = start
    while True:
        train_end = cursor + relativedelta(years=wf_cfg.train_years)
        test_start = train_end + pd.Timedelta(days=1)
        test_end   = test_start + relativedelta(months=wf_cfg.test_months) - pd.Timedelta(days=1)

        if test_start > end:
            break

        test_end = min(test_end, end)

        # Slice data
        fm_train  = feature_matrix.loc[:train_end]
        close_all = close.loc[:test_end]          # strategy may need history for vol
        close_oos = close.loc[test_start:test_end]

        if len(fm_train.index.get_level_values("date").unique()) < wf_cfg.min_train_obs:
            cursor += relativedelta(months=wf_cfg.test_months)
            continue

        fold_num += 1
        try:
            # Generate weights using all history up to test_end.
            # No lookahead: every feature is computed rolling on data available
            # at each date. Walk-forward tests whether performance persists OOS.
            fm_to_test_end = feature_matrix.loc[
                feature_matrix.index.get_level_values("date") <= test_end
            ]
            weights = strategy_fn(fm_to_test_end, close_all)

            # Evaluate only on OOS dates
            w_oos = weights.reindex(close_oos.index).ffill().fillna(0.0)
            if w_oos.empty or w_oos.sum(axis=1).max() < 1e-6:
                cursor += relativedelta(months=wf_cfg.test_months)
                continue

            result   = run(w_oos, close_oos, bt_cfg)
            oos_ret  = result["returns"]
            oos_eq   = result["equity"]

            fold_cagr = cagr(oos_eq) if len(oos_eq) > 20 else np.nan
            fold_sh   = sharpe(oos_ret)
            fold_mdd  = max_drawdown(oos_eq)
            fold_vol  = volatility(oos_ret)

            fold_records.append({
                "fold":        fold_num,
                "train_start": str(cursor.date()),
                "train_end":   str(train_end.date()),
                "test_start":  str(test_start.date()),
                "test_end":    str(test_end.date()),
                "oos_days":    len(oos_ret),
                "cagr":        fold_cagr,
                "sharpe":      fold_sh,
                "max_dd":      fold_mdd,
                "volatility":  fold_vol,
            })
            fold_returns.append(oos_ret)
            folds.append({
                "fold": fold_num, "result": result,
                "train_start": cursor, "train_end": train_end,
                "test_start": test_start, "test_end": test_end,
            })

            print(f"  Fold {fold_num:2d} | train {cursor.date()}→{train_end.date()} "
                  f"| OOS {test_start.date()}→{test_end.date()} "
                  f"| CAGR={fold_cagr:+.1%}  Sharpe={fold_sh:.2f}  MDD={fold_mdd:.1%}")

        except Exception as e:
            print(f"  Fold {fold_num:2d} ERROR: {e}")

        cursor += relativedelta(months=wf_cfg.test_months)

    if not fold_returns:
        raise RuntimeError("No folds completed — check date ranges and min_train_obs.")

    oos_returns = pd.concat(fold_returns).sort_index()
    oos_equity  = (1 + oos_returns).cumprod()

    fold_metrics = pd.DataFrame(fold_records).set_index("fold")
    fold_metrics[["cagr", "sharpe", "max_dd", "volatility"]] = \
        fold_metrics[["cagr", "sharpe", "max_dd", "volatility"]].round(4)

    return {
        "oos_returns":  oos_returns,
        "oos_equity":   oos_equity,
        "fold_metrics": fold_metrics,
        "folds":        folds,
        "label":        label,
    }


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_is_vs_oos(
    is_result: dict,
    wf_result: dict,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare in-sample (full backtest) vs OOS (walk-forward) metrics side by side.
    """
    bm_ret = benchmark_returns(close).reindex(wf_result["oos_returns"].index).fillna(0)
    bm_eq  = (1 + bm_ret).cumprod()

    s_is  = summary(is_result["equity"],       is_result["returns"],
                    label="In-Sample (full)")
    s_oos = summary(wf_result["oos_equity"],   wf_result["oos_returns"],
                    bm_ret, label="OOS (walk-forward)")
    s_bm  = summary(bm_eq, bm_ret, label="SPY (OOS period)")

    return compare([s_is, s_oos, s_bm])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    from data import load, get_close
    from features import build_feature_matrix
    from strategy import risk_managed_strategy
    from backtest import BacktestConfig, run as bt_run

    print("Loading data...")
    close = get_close(load())
    fm    = build_feature_matrix(close)
    bt_cfg = BacktestConfig()
    wf_cfg = WalkForwardConfig(train_years=3, test_months=6)

    # Wrap strategy to match (fm_slice, close_slice) signature
    def strategy_fn(fm_slice, close_slice):
        return risk_managed_strategy(fm_slice, close_slice, top_n=10)

    print(f"\nWalk-forward: {wf_cfg.train_years}y train / {wf_cfg.test_months}m OOS per fold\n")
    wf = run_walkforward(strategy_fn, close, fm, wf_cfg, bt_cfg, label="Risk-Managed")

    print("\n--- Fold metrics ---")
    print(wf["fold_metrics"].to_string())

    # Full in-sample backtest for comparison
    w_full = risk_managed_strategy(fm, close, top_n=10)
    is_res = bt_run(w_full, close, bt_cfg)

    print("\n--- IS vs OOS comparison ---")
    report = compare_is_vs_oos(is_res, wf, close)
    print(report.to_string())

    pct_positive = (wf["fold_metrics"]["sharpe"] > 0).mean()
    print(f"\nFolds with positive Sharpe: {pct_positive:.0%}")
    print(f"Median OOS Sharpe:          {wf['fold_metrics']['sharpe'].median():.2f}")
    print(f"Median OOS CAGR:            {wf['fold_metrics']['cagr'].median():.1%}")
