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
    scale = (target_vol / rolling_vol).clip(0.5, 2.0)  # cap leverage 0.5x–2x
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
    rebal_scores = scores.resample(rebalance_freq).last()

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

    1. Market timing: SPY below SMA200 → cut exposure to 50%
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
        scale = np.clip(1.3 / avg_vol_regime, 0.3, 1.0)
        w = w * scale

        weights_list.append(pd.Series(w, name=date))

    rebal_weights = pd.DataFrame(weights_list).fillna(0.0)
    all_dates = composite_wide.index
    weights = rebal_weights.reindex(all_dates).ffill().fillna(0.0)
    weights = weights.reindex(columns=composite_wide.columns, fill_value=0.0)

    # --- Overlay 1: market timing (daily) ---
    spy_sma200 = close["SPY"].rolling(200).mean()
    spy_above  = (close["SPY"] > spy_sma200).reindex(weights.index)
    # Smooth the signal: require SPY to be below SMA200 for 5 consecutive days
    # to avoid whipsawing on brief dips
    spy_risk_off = (~spy_above).rolling(5, min_periods=1).min().astype(bool)
    market_scale = spy_risk_off.map({True: 0.50, False: 1.0})
    weights = weights.mul(market_scale, axis=0)

    # --- Overlay 2: vol scaling (daily) ---
    weights = vol_scale(weights, close, target_vol=target_vol)

    return weights


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

    print("=== Momentum strategy ===")
    print(w1.tail(3).to_string())
    print("\n=== Composite strategy ===")
    print(w2.tail(3).to_string())
    print("\nWeight sum check (should be ≤1):")
    print(w1.sum(axis=1).describe())
