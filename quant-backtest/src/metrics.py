"""
Performance metrics: Sharpe, Sortino, Calmar, drawdown, alpha/beta, and summary report.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def total_return(equity: pd.Series) -> float:
    return equity.iloc[-1] / equity.iloc[0] - 1.0


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0


def sharpe(returns: pd.Series, risk_free: float = 0.04) -> float:
    """Annualised Sharpe ratio (daily returns, annualised)."""
    rf_daily = (1 + risk_free) ** (1 / 252) - 1
    excess = returns - rf_daily
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(252)


def sortino(returns: pd.Series, risk_free: float = 0.04) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    rf_daily = (1 + risk_free) ** (1 / 252) - 1
    excess = returns - rf_daily
    downside = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return excess.mean() / downside * np.sqrt(252)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak


def calmar(equity: pd.Series) -> float:
    """CAGR / abs(max drawdown)."""
    mdd = abs(max_drawdown(equity))
    return np.nan if mdd == 0 else cagr(equity) / mdd


def volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(252)


def win_rate(returns: pd.Series) -> float:
    return (returns > 0).mean()


def avg_win_loss(returns: pd.Series) -> float:
    """Ratio of average win to average loss magnitude."""
    wins  = returns[returns > 0].mean()
    losses = returns[returns < 0].abs().mean()
    return wins / losses if losses != 0 else np.nan


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (positive number = loss)."""
    return -returns.quantile(1 - confidence)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall."""
    threshold = returns.quantile(1 - confidence)
    tail = returns[returns <= threshold]
    return -tail.mean() if len(tail) > 0 else np.nan


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = 0.04,
) -> tuple[float, float]:
    """Annualised alpha and beta versus benchmark."""
    rf_daily = (1 + risk_free) ** (1 / 252) - 1
    r  = returns - rf_daily
    rb = benchmark_returns - rf_daily

    aligned = pd.concat([r, rb], axis=1).dropna()
    r, rb = aligned.iloc[:, 0], aligned.iloc[:, 1]

    beta = r.cov(rb) / rb.var()
    alpha_daily = r.mean() - beta * rb.mean()
    alpha_ann = (1 + alpha_daily) ** 252 - 1
    return alpha_ann, beta


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if active.std() == 0:
        return np.nan
    return active.mean() / active.std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Drawdown analysis
# ---------------------------------------------------------------------------

def drawdown_table(equity: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Return a table of the top-N worst drawdown periods."""
    dd = drawdown_series(equity)
    in_dd = False
    records = []
    start = None
    peak_val = equity.iloc[0]

    for date, val in equity.items():
        if not in_dd and val < equity[:date].max():
            in_dd = True
            start = date
            peak_val = equity[:date].max()
        elif in_dd and val >= equity[:date].max():
            trough_date = dd[start:date].idxmin()
            trough_dd   = dd[trough_date]
            records.append({
                "start":    start,
                "trough":   trough_date,
                "recovery": date,
                "drawdown": trough_dd,
                "duration_days": (date - start).days,
            })
            in_dd = False

    if in_dd:
        trough_date = dd[start:].idxmin()
        records.append({
            "start":    start,
            "trough":   trough_date,
            "recovery": pd.NaT,
            "drawdown": dd[trough_date],
            "duration_days": (equity.index[-1] - start).days,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("drawdown").head(top_n)
    df["drawdown"] = df["drawdown"].map("{:.2%}".format)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def summary(
    equity: pd.Series,
    returns: pd.Series,
    benchmark_ret: Optional[pd.Series] = None,
    risk_free: float = 0.04,
    label: str = "Strategy",
) -> pd.Series:
    """Return a single-column Series with all key metrics."""
    stats = {
        "Total Return":      f"{total_return(equity):.2%}",
        "CAGR":              f"{cagr(equity):.2%}",
        "Volatility (ann)":  f"{volatility(returns):.2%}",
        "Sharpe Ratio":      f"{sharpe(returns, risk_free):.2f}",
        "Sortino Ratio":     f"{sortino(returns, risk_free):.2f}",
        "Calmar Ratio":      f"{calmar(equity):.2f}",
        "Max Drawdown":      f"{max_drawdown(equity):.2%}",
        "Win Rate":          f"{win_rate(returns):.2%}",
        "Avg Win / Loss":    f"{avg_win_loss(returns):.2f}",
        "VaR 95%  (daily)":  f"{value_at_risk(returns):.2%}",
        "CVaR 95% (daily)":  f"{expected_shortfall(returns):.2%}",
    }

    if benchmark_ret is not None:
        a, b = alpha_beta(returns, benchmark_ret, risk_free)
        ir   = information_ratio(returns, benchmark_ret)
        stats["Alpha (ann)"]  = f"{a:.2%}"
        stats["Beta"]         = f"{b:.2f}"
        stats["Info Ratio"]   = f"{ir:.2f}"

    return pd.Series(stats, name=label)


def compare(summaries: list[pd.Series]) -> pd.DataFrame:
    """Stack multiple summary Series side by side."""
    return pd.concat(summaries, axis=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    from data import load, get_close
    from features import build_feature_matrix
    from strategy import momentum_strategy, composite_strategy, risk_managed_strategy
    from backtest import run, BacktestConfig, benchmark_returns, benchmark_equity

    close = get_close(load())
    fm    = build_feature_matrix(close)
    cfg   = BacktestConfig()

    w1 = momentum_strategy(fm, top_n=10, signal="mom_6m")
    w2 = composite_strategy(fm, top_n=10)
    w3 = risk_managed_strategy(fm, close, top_n=10, target_vol=0.15)

    r1 = run(w1, close, cfg)
    r2 = run(w2, close, cfg)
    r3 = run(w3, close, cfg)

    bm_ret = benchmark_returns(close, "SPY").reindex(r1["returns"].index)
    bm_eq  = benchmark_equity(close).reindex(r1["equity"].index)
    bm_ret2 = bm_eq.pct_change().fillna(0)

    s1   = summary(r1["equity"], r1["returns"], bm_ret, label="Momentum")
    s2   = summary(r2["equity"], r2["returns"], bm_ret, label="Composite")
    s3   = summary(r3["equity"], r3["returns"], bm_ret, label="Risk-Managed")
    s_bm = summary(bm_eq, bm_ret2, label="SPY")

    report = compare([s1, s2, s3, s_bm])
    print("\n" + "=" * 78)
    print(report.to_string())
    print("=" * 78)

    print("\n--- Top drawdowns (Composite) ---")
    print(drawdown_table(r2["equity"]).to_string(index=False))
    print("\n--- Top drawdowns (Risk-Managed) ---")
    print(drawdown_table(r3["equity"]).to_string(index=False))
