"""
Risk module: Value-at-Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall).

Supports four estimation methods:
  - historical  : empirical quantile of the return distribution
  - parametric  : Student-t fit (mean + σ) closed-form
  - monte_carlo : Student-t fit + MC simulation (captures fat tails)
  - garch       : rolling GARCH(1,1) with Student-t innovations

All loss measures are returned as positive numbers (loss convention).
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal

Method = Literal["historical", "parametric", "monte_carlo", "garch"]
_MC_SIMS = 10_000  # default simulation draws — raise to 100k for publication-quality estimates


# ---------------------------------------------------------------------------
# Point-in-time VaR / CVaR
# ---------------------------------------------------------------------------

def var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: Method = "historical",
) -> float:
    """
    Value-at-Risk at the given confidence level.

    Returns the loss threshold exceeded with probability (1 - confidence).
    Positive output means a loss of that magnitude.
    """
    if method == "historical":
        return -returns.quantile(1 - confidence)

    if method == "parametric":
        df_t, loc, scale = stats.t.fit(returns)
        z = stats.t.ppf(1 - confidence, df=df_t)
        return -(loc + z * scale)

    if method == "monte_carlo":
        # Fit a Student-t to capture fat tails, then simulate
        df_t, loc, scale = stats.t.fit(returns)
        rng = np.random.default_rng(42)
        simulated = stats.t.rvs(df_t, loc=loc, scale=scale, size=_MC_SIMS, random_state=rng)
        return -np.quantile(simulated, 1 - confidence)

    raise ValueError(f"Unknown method: {method!r}")


def cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    method: Method = "historical",
) -> float:
    """
    Conditional VaR (Expected Shortfall) — mean loss beyond the VaR threshold.
    Positive output means a loss of that magnitude.
    """
    if method == "historical":
        threshold = returns.quantile(1 - confidence)
        tail = returns[returns < threshold]
        return -tail.mean() if len(tail) > 0 else np.nan

    if method == "parametric":
        df_t, loc, scale = stats.t.fit(returns)
        z = stats.t.ppf(1 - confidence, df=df_t)
        # analytical ES for Gaussian
        es = (stats.t.pdf(z, df=df_t) / (1 - confidence)) * ((df_t + z**2) / (df_t - 1)) * scale
        return -(loc - es)

    if method == "monte_carlo":
        df_t, loc, scale = stats.t.fit(returns)
        rng = np.random.default_rng(42)
        simulated = stats.t.rvs(df_t, loc=loc, scale=scale, size=_MC_SIMS, random_state=rng)
        threshold = np.quantile(simulated, 1 - confidence)
        tail = simulated[simulated <= threshold]
        return -tail.mean() if len(tail) > 0 else np.nan

    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Multi-confidence level table
# ---------------------------------------------------------------------------

def risk_table(
    returns: pd.Series,
    confidences: list[float] = [0.90, 0.95, 0.99],
    methods: list[Method] = ["historical", "parametric", "monte_carlo"],
    annualise: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame with VaR and CVaR for each (confidence, method) pair.

    If annualise=True, scales daily figures by sqrt(252).
    """
    scale = np.sqrt(252) if annualise else 1.0
    rows = []
    for c in confidences:
        for m in methods:
            rows.append({
                "confidence": f"{c:.0%}",
                "method": m,
                "VaR":  var(returns, c, m) * scale,
                "CVaR": cvar(returns, c, m) * scale,
            })
    return pd.DataFrame(rows).set_index(["confidence", "method"])


# ---------------------------------------------------------------------------
# Rolling VaR / CVaR
# ---------------------------------------------------------------------------

def rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: Method = "historical",
) -> pd.Series:
    """Rolling VaR series (positive = loss)."""
    def _var(w):
        return var(w, confidence, method)
    return returns.rolling(window).apply(_var, raw=False)


def rolling_cvar(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: Method = "historical",
) -> pd.Series:
    """Rolling CVaR series (positive = loss)."""
    def _cvar(w):
        return cvar(w, confidence, method)
    return returns.rolling(window).apply(_cvar, raw=False)


# ---------------------------------------------------------------------------
# Multi-strategy comparison helper
# ---------------------------------------------------------------------------

def compare_risk(
    strategy_returns: dict[str, pd.Series],
    confidences: list[float] = [0.95, 0.99],
    method: Method = "historical",
    annualise: bool = False,
) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing VaR and CVaR across strategies.

    Returns a DataFrame indexed by strategy, with columns like
    VaR_95%, CVaR_95%, VaR_99%, CVaR_99%.
    """
    scale = np.sqrt(252) if annualise else 1.0
    rows = {}
    for name, ret in strategy_returns.items():
        row = {}
        for c in confidences:
            tag = f"{c:.0%}"
            row[f"VaR {tag}"]  = var(ret, c, method) * scale
            row[f"CVaR {tag}"] = cvar(ret, c, method) * scale
        rows[name] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Kupiec POF (Proportion of Failures) backtest
# ---------------------------------------------------------------------------

def kupiec_test(
    returns: pd.Series,
    confidence: float = 0.95,
    method: Method = "historical",
    var_series: pd.Series | None = None,
) -> dict:
    """
    Kupiec (1995) Proportion of Failures test for VaR calibration.

    Tests H0: observed breach rate == expected rate (1 - confidence).
    Rejects H0 when VaR is systematically too tight or too loose.

    Parameters
    ----------
    returns     : daily return series
    confidence  : VaR confidence level used during estimation
    method      : estimation method passed to var()
    var_series  : optional pre-computed VaR series (positive = loss);
                  if None, a single in-sample VaR is computed and applied flat

    Returns
    -------
    dict with keys:
        T           – number of observations
        N           – number of breaches
        breach_rate – N / T
        expected    – 1 - confidence
        lr_stat     – likelihood-ratio test statistic
        p_value     – p-value under chi-squared(1)
        reject_h0   – True if p_value < 0.05
    """
    p_expected = 1.0 - confidence

    if var_series is not None:
        aligned = var_series.reindex(returns.index).dropna()
        ret_aligned = returns.reindex(aligned.index)
        breaches = ret_aligned < -aligned
    else:
        v = var(returns, confidence, method)
        breaches = returns < -v

    T = int(breaches.count())
    N = int(breaches.sum())

    if N == 0 or N == T:
        # Edge case: LR is undefined; return nan
        return dict(T=T, N=N, breach_rate=N / T, expected=p_expected,
                    lr_stat=np.nan, p_value=np.nan, reject_h0=False)

    p_obs = N / T
    # LR statistic: -2 * ln(L0 / L1)
    lr = -2 * (
        N * np.log(p_expected / p_obs)
        + (T - N) * np.log((1 - p_expected) / (1 - p_obs))
    )
    p_value = float(stats.chi2.sf(lr, df=1))

    return dict(
        T=T,
        N=N,
        breach_rate=p_obs,
        expected=p_expected,
        lr_stat=round(lr, 4),
        p_value=round(p_value, 4),
        reject_h0=p_value < 0.05,
    )


def rolling_var_series(
    returns: pd.Series,
    confidence: float = 0.95,
    method: Method = "historical",
    window: int = 252,
) -> pd.Series:
    """
    Rolling out-of-sample VaR series.
    VaR at day T is estimated on returns [T-window : T-1], then tested against return at T.

    historical  — vectorised via pandas rolling quantile (fast)
    parametric  — vectorised via rolling mean/std; df_t fitted once on the full
                  series (stable across windows, avoids per-step MLE) (fast)
    monte_carlo — Python loop with per-step Student-t fit + simulation (slow)
    """
    if method == "historical":
        # shift(1) ensures the quantile is computed on [T-window : T-1]
        return (
            returns
            .shift(1)
            .rolling(window)
            .quantile(1 - confidence)
            .mul(-1)
            .dropna()
        )

    if method == "parametric":
        # df_t is a shape parameter — fit once, reuse across all windows
        df_t, _, _ = stats.t.fit(returns)
        z = stats.t.ppf(1 - confidence, df=df_t)
        roll = returns.shift(1).rolling(window)
        loc   = roll.mean()
        scale = roll.std()
        result = -(loc + z * scale)
        return result.dropna()

    if method == "garch":
        return garch_var_series(returns, confidence, window)

    # monte_carlo: no shortcut — fit + simulate per step
    var_values = {}
    for i in range(window, len(returns)):
        train = returns.iloc[i - window:i]
        var_values[returns.index[i]] = var(train, confidence, method)
    return pd.Series(var_values)


# ---------------------------------------------------------------------------
# GARCH-based rolling VaR
# ---------------------------------------------------------------------------

def garch_var_series(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
    p: int = 1,
    q: int = 1,
) -> pd.Series:
    """
    Rolling GARCH(p,q)-based VaR series.

    At each time step t, fits GARCH on returns[t-window : t-1] and
    forecasts the one-step-ahead conditional volatility sigma_t.
    VaR is computed as:

        VaR_t = -(mu + sigma_t * z_alpha)

    where z_alpha is the quantile of the fitted Student-t distribution
    and mu is the in-window mean return.

    On convergence failure, falls back to historical VaR for that window.

    Parameters
    ----------
    returns    : daily return series
    confidence : VaR confidence level
    window     : rolling estimation window in days
    p, q       : GARCH lag orders
    """
    from arch import arch_model

    total = len(returns) - window
    var_values = {}

    for i in range(window, len(returns)):
        if (i - window) % 250 == 0:
            print(f"GARCH fitting: {i - window}/{total} steps", end="\r")

        train = returns.iloc[i - window:i]
        date  = returns.index[i]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(
                    train * 100,
                    vol="Garch", p=p, q=q, dist="t", mean="Constant",
                )
                res = model.fit(
                    disp="off", options={"maxiter": 200}
                )

            forecast = res.forecast(horizon=1, reindex=False)
            sigma = np.sqrt(forecast.variance.values[-1, 0]) / 100

            # Sanity check: degenerate GARCH solutions (convergence to wrong optimum)
            # produce sigma orders of magnitude above historical vol without raising.
            # Fall back to historical if sigma > 5× the window's empirical std.
            if sigma > train.std() * 5:
                raise ValueError("degenerate sigma")

            nu  = res.params.get("nu", res.params.iloc[-1])
            mu  = train.mean()
            z   = stats.t.ppf(1 - confidence, df=nu)
            var_values[date] = -(mu + sigma * z)

        except Exception:
            var_values[date] = var(train, confidence, method="historical")

    print()  # newline after progress indicator
    return pd.Series(var_values)


def kupiec_table(
    strategy_returns: dict[str, pd.Series],
    confidences: list[float] = [0.90, 0.95, 0.99],
    method: Method = "historical",
    window: int = 252,
) -> pd.DataFrame:
    """
    Run the Kupiec test for every (strategy, confidence) combination and
    return a summary DataFrame.

    Uses a rolling out-of-sample VaR so breach counting is genuinely OOS.
    """
    rows = []
    for name, ret in strategy_returns.items():
        for c in confidences:
            var_s = rolling_var_series(ret, c, method, window)
            res = kupiec_test(ret, c, method, var_series=var_s)
            rows.append({
                "Strategy":    name,
                "Confidence":  f"{c:.0%}",
                "T":           res["T"],
                "Breaches":    res["N"],
                "Breach Rate": f"{res['breach_rate']:.2%}",
                "Expected":    f"{res['expected']:.2%}",
                "LR Stat":     res["lr_stat"],
                "p-value":     res["p_value"],
                "Reject H0":   res["reject_h0"],
            })
    return pd.DataFrame(rows).set_index(["Strategy", "Confidence"])
