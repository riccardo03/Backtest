"""
Data pipeline: download, clean, and store OHLCV data via yfinance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

UNIVERSE: dict[str, list[str]] = {
    "etf": [
        "SPY",   # S&P 500
        "QQQ",   # Nasdaq 100
        "IWM",   # Russell 2000
        "EFA",   # Developed ex-US
        "EEM",   # Emerging markets
        "TLT",   # 20yr Treasuries
        "GLD",   # Gold
        "VNQ",   # REITs
    ],
    "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA"],
    "finance": ["JPM", "BAC", "GS"],
    "health": ["JNJ", "UNH"],
    "energy": ["XOM", "CVX"],
    "consumer": ["WMT", "HD", "PG", "KO"],
    "industrial": ["CAT", "BA"],
    "media": ["DIS", "NFLX"],
}

ALL_TICKERS: list[str] = [t for group in UNIVERSE.values() for t in group]

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(
    tickers: Optional[list[str]] = None,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted close prices (and full OHLCV) for the given tickers.
    Returns a MultiIndex DataFrame: (field, ticker).
    """
    tickers = tickers or ALL_TICKERS
    log.info("Downloading %d tickers from %s to %s ...", len(tickers), start, end or "today")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        df = raw
    else:
        # single ticker — add ticker level
        df = pd.concat({tickers[0]: raw}, axis=1).swaplevel(axis=1)

    log.info("Raw shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame, max_missing_pct: float = 0.05) -> pd.DataFrame:
    """
    Drop tickers with too many missing values, forward-fill small gaps,
    and trim to dates where at least 90% of tickers have data.
    """
    close = df["Close"]

    # Drop tickers with > max_missing_pct NaNs
    missing = close.isna().mean()
    bad = missing[missing > max_missing_pct].index.tolist()
    if bad:
        log.warning("Dropping tickers with >%.0f%% missing data: %s", max_missing_pct * 100, bad)
        df = df.drop(columns=bad, level=1)

    # Forward-fill up to 5 days (holidays, halts)
    df = df.ffill(limit=5)

    # Trim leading NaNs: keep dates where ≥90% of close prices are present
    close = df["Close"]
    enough = close.notna().mean(axis=1) >= 0.90
    df = df.loc[enough]

    log.info("Clean shape: %s  |  tickers: %d", df.shape, df["Close"].shape[1])
    return df


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, name: str = "universe") -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(path)
    log.info("Saved → %s", path)
    return path


def load(name: str = "universe") -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run download() first")
    df = pd.read_parquet(path)
    log.info("Loaded %s  shape: %s", path.name, df.shape)
    return df


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_close(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return a plain (date × ticker) DataFrame of adjusted close prices."""
    if df is None:
        df = load()
    return df["Close"]


def get_returns(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Daily log returns from adjusted close."""
    close = get_close(df)
    return close.pct_change().dropna(how="all")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = download(start="2015-01-01")
    cleaned = clean(raw)
    save(cleaned)
    print(get_returns(cleaned).tail())
