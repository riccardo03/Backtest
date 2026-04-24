# Quant Backtest — Systematic Equity Strategies

A modular backtesting framework for cross-sectional equity strategies, built on top of momentum signals, composite scoring, and dynamic risk overlays. The system covers the full pipeline from raw price data to walk-forward validated performance metrics.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Strategies](#strategies)
- [Results](#results)
- [Walk-Forward Validation](#walk-forward-validation)
- [Risk & Cost Assumptions](#risk--cost-assumptions)
- [Limitations & Known Issues](#limitations--known-issues)
- [Setup](#setup)

---

## Overview

The project implements and compares three increasingly sophisticated equity strategies on a universe of ~140 US large-cap assets (2016–2026), evaluated with strict walk-forward out-of-sample validation to prevent overfitting.

The primary goal is **not** to maximise raw CAGR, but to achieve a better risk-adjusted profile than a passive SPY benchmark — specifically a higher Calmar ratio and materially lower drawdowns.

---

## Project Structure

```
quant-backtest/
├── src/
│   ├── data.py          # Data loading and cleaning pipeline
│   ├── features.py      # Feature engineering (momentum, vol, z-scores)
│   ├── strategy.py      # Strategy implementations (weights generation)
│   ├── backtest.py      # Backtest engine with commissions and slippage
│   ├── metrics.py       # Performance metrics (Sharpe, Calmar, CVaR, etc.)
│   └── walkforward.py   # Walk-forward validation engine
├── notebooks/
│   └── analysis.ipynb   # Equity curves, drawdowns, heatmaps, WF analysis
├── data/
│   └── universe.parquet # Cached OHLCV data (145 assets, 2842 dates)
├── requirements.txt
└── README.md
```

---

## Strategies

### 1. Momentum (Baseline)

Each month-end, ranks all assets by 6-month momentum (`mom_6m`) and goes long the top 10. Weights are equal across selected assets, forward-filled to daily frequency between rebalances.

This is the baseline. Any more complex strategy should beat it on a risk-adjusted basis to justify the added complexity.

### 2. Composite

Combines four signals into a weighted composite score, each cross-sectionally z-scored before aggregation to ensure comparable scale:

| Signal | Weight | Rationale |
|---|---|---|
| `mom_6m` | 35% | Core momentum |
| `mom_12_1` | 25% | 12-month momentum excluding last month (avoids short-term reversal) |
| `sma_cross` | 20% | Price above/below moving average — trend confirmation |
| `dist_52w_high` | 20% | Distance from 52-week high — breakout signal |

Adds a **vol-regime filter**: when market stress is elevated (`vol_regime > 1.3`), position sizes are scaled down proportionally via `clip(1.3 / avg_vol_regime, 0.3, 1.0)`.

### 3. Risk-Managed

Builds on the composite strategy with three additional risk overlays:

**Rebalance-level (monthly):**
- Defensive tilt: subtracts 10% of `zscore_20` from the composite score, penalising assets extended far above their recent mean (mean-reversion risk)

**Daily overlays (react faster than monthly rebalancing):**
- **Market timing:** when SPY remains below its 200-day SMA for 5+ consecutive days, exposure is scaled to 50%. The 5-day smoothing prevents whipsawing on brief dips.
- **Vol scaling:** dynamically sizes the portfolio to target 15% annualised volatility using 21-day realised portfolio vol, with leverage capped at 0.5x–2x.

The order matters: market timing is applied first, vol scaling operates on the already-adjusted weights.

---

## Results

All metrics computed on out-of-sample walk-forward periods (2019–2026).

| Metric | Risk-Managed (OOS) | SPY |
|---|---|---|
| CAGR | 13.30% | 16.86% |
| Volatility (ann) | 17.20% | 19.54% |
| Sharpe Ratio | 0.59 | 0.70 |
| Sortino Ratio | 0.82 | 0.86 |
| **Calmar Ratio** | **0.70** | **0.50** |
| **Max Drawdown** | **−18.98%** | **−33.72%** |
| VaR 95% (daily) | 1.70% | 1.75% |
| CVaR 95% (daily) | 2.47% | 2.94% |
| Beta | 0.65 | 1.00 |
| Alpha (ann) | 1.13% | — |

**Key takeaway:** the strategy does not outperform SPY on raw CAGR — a direct consequence of the low beta (0.65). In a sustained bull market, lower market exposure means lower absolute returns. The value is in the risk profile: max drawdown is reduced by ~44% relative to SPY, and the Calmar ratio is 40% higher, meaning more return per unit of maximum loss tolerated.

The Sortino (0.82) exceeding the Sharpe (0.59) indicates the return distribution is positively skewed — upside volatility is larger than downside volatility, which is the desirable asymmetry.

---

## Walk-Forward Validation

**Setup:** 3-year rolling training window, 6-month OOS evaluation period, 15 folds (2016–2026).

| Summary | Value |
|---|---|
| Folds with positive Sharpe | 80% (12/15) |
| Median OOS Sharpe | 0.45 |
| Median OOS CAGR | 10.4% |
| IS → OOS Sharpe degradation | 0.75 → 0.59 (−21%) |

The IS→OOS degradation of 21% is moderate and consistent with a system that is not overfitted. If degradation were near zero, the walk-forward would be a formality; if it were 60–80%, it would signal significant overfitting on the training set.

**Per-fold breakdown:**

| Fold | OOS Period | CAGR | Sharpe | MDD | Notes |
|---|---|---|---|---|---|
| 1 | 2019 H1 | +8.0% | 0.33 | −10.7% | Quiet market post Q4 2018 correction |
| 2 | 2019 H2 | −7.3% | −0.63 | −10.5% | US-China trade tensions, choppy market |
| 3 | 2020 H1 | +47.6% | 1.41 | −19.0% | Covid crash + recovery. MDD reflects March 2020 |
| 4 | 2020 H2 | +38.3% | 1.78 | −7.9% | Post-Covid bull run, ideal momentum conditions |
| 5 | 2021 H1 | +12.5% | 0.55 | −7.4% | Normal |
| 6 | 2021 H2 | +12.9% | 0.55 | −8.3% | Normal |
| 7 | 2022 H1 | −22.5% | −1.77 | −14.1% | Worst fold. Fed tightening, momentum crash (growth→value rotation) |
| 8 | 2022 H2 | +9.8% | 0.42 | −9.5% | Recovery |
| 9 | 2023 H1 | +10.5% | 0.45 | −12.0% | Normal |
| 10 | 2023 H2 | +6.8% | 0.24 | −10.4% | Weak |
| 11 | 2024 H1 | +74.4% | 3.29 | −7.9% | See note below |
| 12 | 2024 H2 | +18.1% | 0.82 | −9.9% | Good |
| 13 | 2025 H1 | −0.9% | −0.21 | −17.2% | Slightly negative |
| 14 | 2025 H2 | +14.5% | 0.61 | −5.4% | Good, minimal drawdown |
| 15 | 2026 H1* | +6.0% | 0.20 | −9.9% | Partial fold (75 days) |

**Note on Fold 11 (Sharpe 3.29):** this fold is anomalously strong and requires honest interpretation. Contribution analysis shows the performance was predominantly driven by NVDA, which returned ~+80% in H1 2024 on the back of the AI infrastructure boom. The momentum signal correctly identified NVDA as a top-ranked asset at the start of the period (strong 6m and 12m momentum as of end-2023 training data), and the system held it with a significant weight throughout. This is not a statistical artefact or look-ahead bias — the signal worked as intended. However, a Sharpe of 3.29 reflects a single concentrated event, not a repeatable edge. The median OOS Sharpe of 0.45 (which is robust to this outlier) is the more representative expectation.

**Worst fold (Fold 7 — 2022 H1):** the strategy held high-momentum growth stocks entering 2022, exactly the names hit hardest by the Fed's hawkish pivot. This is the canonical momentum crash: a regime change (low-rate → high-rate) causes rapid factor rotation that punishes recent winners. The market timing overlay (SPY vs SMA200) partially mitigated the damage but could not prevent it entirely.

---

## Risk & Cost Assumptions

| Parameter | Value | Notes |
|---|---|---|
| Commission | 0.10% per trade | Conservative estimate for retail on Interactive Brokers large-caps |
| Slippage | 0.05% per trade | Realistic for liquid US large-caps (>$500M ADV) |
| Execution | Next open after signal | Avoids look-ahead bias on close prices |
| Rebalancing | Month-end | Risk overlays applied daily |
| Max leverage | 2.0x | Hard cap in vol scaling (`clip(0.5, 2.0)`) |

Transaction costs are simulated as `turnover × 2 × commission_rate`. At median monthly one-way turnover of ~19% (Momentum) and ~30% (Composite), annual cost drag is approximately 45–70 bps — material but not dominant.

The capacity check at AUM = $10M flags 7 positions above $500k notional (CAT, CVX, GLD, GOOGL, JNJ, WMT, XOM). All are investment-grade large-caps with daily volumes in the billions — no liquidity concern at this scale.

---

## Limitations & Known Issues

**What this system does not handle:**

- **No live execution.** The notebook produces target weights; it does not interface with any broker API. Converting to a live system requires an execution layer (e.g., Alpaca or IBKR API) and a daily scheduler.
- **EOD data only.** All signals are computed on end-of-day prices. Intraday dynamics, opening gaps, and overnight risk are not modelled.
- **No corporate actions adjustment beyond split/dividend.** M&A events, delistings, and index reconstitutions are handled passively by `yfinance` data.
- **Turnover bug (composite strategy):** max one-way turnover exceeds 100% in some months, which is theoretically impossible and indicates an accounting issue in the turnover calculation when portfolio weights are not normalised to 1. Under investigation.
- **Sector concentration:** the strategy has no explicit sector constraints. In periods of strong factor concentration (e.g., AI/tech in 2024), the portfolio can implicitly become a sector bet. A sector exposure breakdown over time is a planned addition.

---

## Setup

```bash
git clone https://github.com/riccardo03/Backtest.git
cd Backtest
pip install -r requirements.txt
```

**Run the analysis notebook:**

```bash
jupyter notebook notebooks/analysis.ipynb
```

**Run strategies from the command line:**
