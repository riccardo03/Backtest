"""
Feature engineering: momentum, volatility, mean-reversion, and volume signals.
All functions take a (date × ticker) close DataFrame and return same-shape DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def momentum(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Total return over the last `window` trading days."""
    return close.pct_change(window)


def momentum_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """Classic Jegadeesh-Titman: 12-month return skipping last month."""
    ret_12 = close.shift(21).pct_change(252 - 21)
    return ret_12


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index (0–100)."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Trend / Moving averages
# ---------------------------------------------------------------------------

def sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return close.rolling(window).mean()


def ema(close: pd.DataFrame, span: int) -> pd.DataFrame:
    return close.ewm(span=span, adjust=False).mean()


def sma_crossover(close: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """+1 when fast SMA > slow SMA, -1 otherwise."""
    signal = (sma(close, fast) > sma(close, slow)).astype(int) * 2 - 1
    return signal.astype(float)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def realized_vol(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualised realised volatility over rolling window."""
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def vol_regime(close: pd.DataFrame, short: int = 21, long: int = 63) -> pd.DataFrame:
    """Ratio of short-term vol to long-term vol. >1 means elevated risk."""
    return realized_vol(close, short) / realized_vol(close, long)


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def zscore(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Z-score of price relative to its rolling mean."""
    mu = close.rolling(window).mean()
    sigma = close.rolling(window).std()
    return (close - mu) / sigma.replace(0, np.nan)


def distance_from_52w_high(close: pd.DataFrame) -> pd.DataFrame:
    """How far (%) each asset is from its 52-week high — used as a momentum proxy."""
    high_52 = close.rolling(252).max()
    return (close - high_52) / high_52


# ---------------------------------------------------------------------------
# Combined feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(close: pd.DataFrame) -> pd.DataFrame:
    """
    Stack all features into a long-format DataFrame:
    index=(date, ticker), columns=feature names.
    """
    features = {
        "mom_1m":        momentum(close, 21),
        "mom_3m":        momentum(close, 63),
        "mom_6m":        momentum(close, 126),
        "mom_12_1":      momentum_12_1(close),
        "rsi_14":        rsi(close, 14),
        "sma_cross":     sma_crossover(close, 50, 200),
        "vol_21":        realized_vol(close, 21),
        "vol_regime":    vol_regime(close, 21, 63),
        "zscore_20":     zscore(close, 20),
        "dist_52w_high": distance_from_52w_high(close),
    }

    frames = []
    for name, df in features.items():
        s = df.stack(future_stack=True)
        s.name = name
        frames.append(s)

    matrix = pd.concat(frames, axis=1)
    matrix.index.names = ["date", "ticker"]

    before = len(matrix)
    matrix = matrix.dropna()
    dropped = before - len(matrix)
    print(f"build_feature_matrix: dropped {dropped} rows with NaN ({dropped/before:.1%}) — warm-up period")

    return matrix


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data import load, get_close
    close = get_close(load())
    fm = build_feature_matrix(close)
    print(fm.tail(10).to_string())
    print("\nShape:", fm.shape)
    print("NaN %:", fm.isna().mean().round(3).to_string())
