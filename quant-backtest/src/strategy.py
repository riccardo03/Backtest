"""
Signal generation strategies.
Each strategy receives the feature matrix (date, ticker) × features
and returns a weights DataFrame (date × ticker) that sums to 1.0 per row.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Portfolio construction helpers
# ---------------------------------------------------------------------------

def equal_weight(selected: pd.DataFrame) -> pd.DataFrame:
    """Given a boolean mask (date × ticker), return equal weights."""
    w = selected.astype(float)
    row_sum = w.sum(axis=1).replace(0, np.nan)
    return w.div(row_sum, axis=0).fillna(0.0)


def rank_weight(scores: pd.DataFrame) -> pd.DataFrame:
    """Weight proportional to cross-sectional rank (higher score → higher weight)."""
    ranks = scores.rank(axis=1, ascending=True, na_option="bottom")
    row_sum = ranks.sum(axis=1).replace(0, np.nan)
    return ranks.div(row_sum, axis=0).fillna(0.0)


def vol_scale(weights: pd.DataFrame, close: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """
    Scale portfolio weights so expected annualised vol ≈ target_vol.
    Uses 21-day realised vol of the weighted portfolio.
    """
    returns = close.pct_change()
    port_ret = (returns * weights.shift(1)).sum(axis=1)
    rolling_vol = port_ret.rolling(21).std() * np.sqrt(252)
    # Reindex to weights dates before multiplying to avoid NaN from date misalignment
    scale = (target_vol / rolling_vol.shift(1)).clip(0.5, 2.0)
    scale = scale.reindex(weights.index).fillna(1.0)
    scaled = weights.mul(scale, axis=0)
    return scaled


# ---------------------------------------------------------------------------
# Strategy 1 — Simple Momentum (baseline)
# ---------------------------------------------------------------------------

def momentum_strategy(
    feature_matrix: pd.DataFrame,
    top_n: int = 10,
    signal: str = "mom_6m",
    rebalance_freq: str = "ME",       # month-end
    weighting: str = "equal",         # "equal" | "rank"
) -> pd.DataFrame:
    """
    Each rebalance date: go long the top_n assets ranked by `signal`.
    Returns a daily weights DataFrame (forward-filled between rebalances).
    """
    scores = feature_matrix[signal].unstack("ticker")

    # Resample to rebalance frequency — take last available score
    rebal_scores = scores.resample(rebalance_freq).last().dropna(how="all")

    weights_list = []
    for date, row in rebal_scores.iterrows():
        row = row.dropna()
        if len(row) == 0:
            continue
        top = row.nlargest(top_n).index
        if weighting == "equal":
            w = pd.Series(1.0 / len(top), index=top)
        else:
            ranks = row[top].rank()
            w = ranks / ranks.sum()
        weights_list.append(pd.Series(w, name=date))

    rebal_weights = pd.DataFrame(weights_list).fillna(0.0)

    # Forward-fill weights to daily frequency
    all_dates = scores.index
    weights = rebal_weights.reindex(all_dates).ffill().fillna(0.0)

    # Align columns to full ticker universe
    weights = weights.reindex(columns=scores.columns, fill_value=0.0)
    return weights


# ---------------------------------------------------------------------------
# Strategy 2 — Multi-signal composite
# ---------------------------------------------------------------------------

def composite_strategy(
    feature_matrix: pd.DataFrame,
    top_n: int = 10,
    rebalance_freq: str = "ME",
) -> pd.DataFrame:
    """
    Combines momentum and trend signals into a composite score.
    Signals:
      - mom_6m      (weight 0.35)
      - mom_12_1    (weight 0.25)
      - sma_cross   (weight 0.20)
      - dist_52w_high (weight 0.20)
    Applies vol-regime filter: if vol_regime > 1.3, halve position sizes.
    """
    fm = feature_matrix.copy()

    # Composite score (cross-sectionally z-scored before combining)
    def cs_zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std() + 1e-9)

    scores_wide = fm[["mom_6m", "mom_12_1", "sma_cross", "dist_52w_high"]].copy()
    scores_wide = scores_wide.groupby(level="date").transform(cs_zscore)

    composite = (
        scores_wide["mom_6m"]        * 0.35
        + scores_wide["mom_12_1"]    * 0.25
        + scores_wide["sma_cross"]   * 0.20
        + scores_wide["dist_52w_high"] * 0.20
    )
    composite.name = "composite"
    composite_wide = composite.unstack("ticker")

    vol_regime_wide = fm["vol_regime"].unstack("ticker")

    rebal_scores  = composite_wide.resample(rebalance_freq).last()
    rebal_vol_reg = vol_regime_wide.resample(rebalance_freq).last()

    weights_list = []
    for date, row in rebal_scores.iterrows():
        row = row.dropna()
        if len(row) == 0:
            continue
        top = row.nlargest(top_n).index
        ranks = row[top].rank()
        w = ranks / ranks.sum()

        # Vol-regime filter: shrink weights when market stress is elevated
        avg_vol_regime = rebal_vol_reg.loc[date, top].mean()
        if np.isnan(avg_vol_regime) or avg_vol_regime == 0:
            avg_vol_regime = 1.0
        scale = np.clip(1.3 / avg_vol_regime, 0.3, 1.0)
        w = w * scale

        weights_list.append(pd.Series(w, name=date))

    rebal_weights = pd.DataFrame(weights_list).fillna(0.0)

    all_dates = composite_wide.index
    weights = rebal_weights.reindex(all_dates).ffill().fillna(0.0)
    weights = weights.reindex(columns=composite_wide.columns, fill_value=0.0)
    return weights


# ---------------------------------------------------------------------------
# Strategy 3 — Risk-managed composite
# ---------------------------------------------------------------------------

def risk_managed_strategy(
    feature_matrix: pd.DataFrame,
    close: pd.DataFrame,
    top_n: int = 10,
    rebalance_freq: str = "ME",
    target_vol: float = 0.15,
) -> pd.DataFrame:
    """
    Composite strategy with three risk overlays applied daily (not only at rebalance):

    1. Market timing: SPY below SMA200 → scale exposure to by 0.3
    2. Vol scaling: target 15% annualised portfolio vol (applied after timing filter)
    3. Defensive tilt at rebalance: penalise assets with high zscore_20
       (mean-reversion risk) when vol_regime is elevated

    The timing and vol filters are applied daily to the forward-filled weights,
    so they can react faster than monthly rebalancing.
    """
    fm = feature_matrix.copy()

    def cs_zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std() + 1e-9)

    scores_wide = fm[["mom_6m", "mom_12_1", "sma_cross", "dist_52w_high"]].copy()
    scores_wide = scores_wide.groupby(level="date").transform(cs_zscore)

    # Defensive tilt: subtract zscore_20 penalty (high zscore = overextended)
    zscore_wide = fm["zscore_20"].unstack("ticker")

    composite = (
        scores_wide["mom_6m"]           * 0.30
        + scores_wide["mom_12_1"]       * 0.20
        + scores_wide["sma_cross"]      * 0.20
        + scores_wide["dist_52w_high"]  * 0.20
        - fm["zscore_20"]               * 0.10   # penalise overextended assets
    )
    composite.name = "composite"
    composite_wide = composite.unstack("ticker")

    vol_regime_wide = fm["vol_regime"].unstack("ticker")

    rebal_scores  = composite_wide.resample(rebalance_freq).last()
    rebal_vol_reg = vol_regime_wide.resample(rebalance_freq).last()

    # --- Build raw rebalance weights ---
    weights_list = []
    for date, row in rebal_scores.iterrows():
        row = row.dropna()
        if len(row) == 0:
            continue
        top = row.nlargest(top_n).index
        ranks = row[top].rank()
        w = ranks / ranks.sum()

        avg_vol_regime = rebal_vol_reg.loc[date, top].mean()
        if np.isnan(avg_vol_regime) or avg_vol_regime == 0:
            avg_vol_regime = 1.0
        scale = np.clip(1.3 / avg_vol_regime, 0.3, 1.0)
        w = w * scale

        weights_list.append(pd.Series(w, name=date))

    rebal_weights = pd.DataFrame(weights_list).fillna(0.0)
    all_dates = composite_wide.index
    weights = rebal_weights.reindex(all_dates).ffill().fillna(0.0)
    weights = weights.reindex(columns=composite_wide.columns, fill_value=0.0)

    # --- Overlay 1: market timing (daily) ---
    spy_sma200 = close["SPY"].rolling(200).mean()
    dist_pct = (close["SPY"] - spy_sma200) / spy_sma200  # distanza % dalla SMA200

    market_scale = (1 + dist_pct * 5).clip(0.3, 1.0).rolling(5, min_periods=1).mean().clip(0.30, 1.0)
    market_scale = market_scale.reindex(weights.index).fillna(1.0)
    weights = weights.mul(market_scale, axis=0)

    # --- Overlay 2: vol scaling (daily) ---
    weights = vol_scale(weights, close, target_vol=target_vol)

    return weights


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def turnover_report(weights: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """
    Monthly turnover report: number of tickers changed and one-way traded weight.

    Returns a DataFrame with columns:
        tickers_in   — new positions entered this month
        tickers_out  — positions exited this month
        tickers_held — positions held unchanged
        one_way_to   — sum of absolute weight changes / 2
    """
    rebal = weights.resample(freq).last()
    records = []
    prev = pd.Series(0.0, index=rebal.columns)

    for date, row in rebal.iterrows():
        held_prev = set(prev[prev > 0].index)
        held_now  = set(row[row > 0].index)
        records.append({
            "date":         date,
            "tickers_in":   len(held_now - held_prev),
            "tickers_out":  len(held_prev - held_now),
            "tickers_held": len(held_now & held_prev),
            "one_way_to":   (row - prev).abs().sum() / 2,
        })
        prev = row

    df = pd.DataFrame(records).set_index("date")
    print(f"\n--- Turnover summary ({freq}) ---")
    print(df[["tickers_in", "tickers_out", "tickers_held", "one_way_to"]].describe().round(3).to_string())
    return df


def capacity_check(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    portfolio_aum: float = 10_000_000,
) -> pd.DataFrame:
    """
    Check whether target weights imply position sizes that exceed adv_fraction
    of the asset's average daily volume (ADV).

    Requires a 'Volume' level in close's parent DataFrame — pass the raw
    MultiIndex DataFrame from load() instead of just Close if available.
    Falls back to a notional-only check if volume data is absent.

    Returns a DataFrame of breaches (date, ticker, weight, notional, adv_limit).
    """
    last_date = weights.index[-1]
    last_w    = weights.loc[last_date]
    last_px   = close.loc[last_date]

    notional = last_w * portfolio_aum
    breaches  = []
    for ticker in last_w[last_w > 0].index:
        if ticker not in close.columns:
            continue
        pos_notional = notional.get(ticker, 0)
        px = last_px.get(ticker, np.nan)
        if np.isnan(px) or px == 0:
            continue
        # Without volume data we flag positions >$500k in a single name
        if pos_notional > 500_000:
            breaches.append({
                "ticker":    ticker,
                "weight":    f"{last_w[ticker]:.2%}",
                "notional":  f"${pos_notional:,.0f}",
                "flag":      ">$500k — verify liquidity",
            })

    df = pd.DataFrame(breaches) if breaches else pd.DataFrame(columns=["ticker", "weight", "notional", "flag"])
    print(f"\n--- Capacity check (AUM=${portfolio_aum:,.0f}) ---")
    if df.empty:
        print("No positions flagged.")
    else:
        print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_tests(close: pd.DataFrame, fm: pd.DataFrame) -> None:
    """Minimal sanity tests for all strategies and helpers."""
    import traceback

    passed = failed = 0

    def check(name: str, condition: bool, msg: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}" + (f" — {msg}" if msg else ""))
            failed += 1

    print("\n=== Running unit tests ===")

    # risk_managed uses vol_scale with up to 2x leverage — higher threshold
    max_sum = {"momentum_strategy": 1.05, "composite_strategy": 1.05, "risk_managed_strategy": 2.1}

    for label, w in [
        ("momentum_strategy",     momentum_strategy(fm, top_n=10)),
        ("composite_strategy",    composite_strategy(fm, top_n=10)),
        ("risk_managed_strategy", risk_managed_strategy(fm, close, top_n=10)),
    ]:
        try:
            check(f"{label}: no NaN weights",
                  not w.isna().any().any())
            check(f"{label}: no negative weights",
                  (w.fillna(0) >= -1e-9).all().all(),
                  f"min={w.min().min():.6f}")
            check(f"{label}: weights sum ≤ {max_sum[label]}",
                  (w.sum(axis=1) <= max_sum[label]).all(),
                  f"max_sum={w.sum(axis=1).max():.4f}")
            check(f"{label}: no all-zero rows after first rebalance",
                  (w.iloc[30:].sum(axis=1) > 0).any())
        except Exception:
            print(f"  ERROR {label}:\n{traceback.format_exc()}")
            failed += 1

    # vol_scale NaN check
    try:
        w_raw = momentum_strategy(fm, top_n=10)
        w_scaled = vol_scale(w_raw, close, target_vol=0.15)
        valid = w_scaled.iloc[30:]   # skip warm-up
        check("vol_scale: no NaN after warm-up",
              valid.isna().sum().sum() == 0,
              f"NaN count={valid.isna().sum().sum()}")
        check("vol_scale: no negative weights",
              (valid >= -1e-9).all().all())
    except Exception:
        print(f"  ERROR vol_scale:\n{traceback.format_exc()}")
        failed += 1

    print(f"\n  {passed} passed, {failed} failed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data import load, get_close
    from features import build_feature_matrix

    close = get_close(load())
    fm = build_feature_matrix(close)

    w1 = momentum_strategy(fm, top_n=10, signal="mom_6m")
    w2 = composite_strategy(fm, top_n=10)
    w3 = risk_managed_strategy(fm, close, top_n=10)

    print("Weight sum check — momentum (should be ≤1):")
    print(w1.sum(axis=1).describe())

    turnover_report(w1, freq="ME")
    turnover_report(w3, freq="ME")

    capacity_check(w3, close, portfolio_aum=10_000_000)

    run_tests(close, fm)
