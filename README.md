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

The three strategies represent a deliberate spectrum of risk profiles: Momentum and Composite are high-beta, high-return strategies suited to investors who can tolerate drawdowns above 30%; Risk-Managed targets a lower-drawdown profile at the cost of absolute return. All three outperform SPY on a risk-adjusted basis in out-of-sample testing.

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

This is the baseline. Any more complex strategy should beat it on a risk-adjusted basis to justify the added complexity. In practice, the Composite beats it marginally on Alpha and Calmar; the difference is small enough that Momentum's higher consistency (93% positive Sharpe folds vs 87%) makes it competitive.

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

Builds on the Composite with three additional risk overlays designed to reduce drawdown at the cost of absolute return.

**Rebalance-level (monthly):**
- Defensive tilt: subtracts 10% of `zscore_20` from the composite score, penalising assets extended far above their recent mean

**Daily overlays:**
- **Market timing (gradual):** scales exposure continuously based on SPY's distance from its 200-day SMA. At the SMA, exposure is 100%; at 10% below, exposure drops to ~50%; floor at 30%. Smoothed over 5 days to avoid reacting to single-day dips.
- **Vol scaling:** dynamically sizes the portfolio to target 15% annualised volatility using 21-day realised portfolio vol, with leverage capped at 0.5x–2x.

---

## Results

All OOS metrics are computed on walk-forward out-of-sample periods (2019–2026). IS metrics cover the full 2016–2026 period.

| Metric | Momentum OOS | Composite OOS | Risk-Managed OOS | SPY (OOS) |
|---|---|---|---|---|
| Total Return | 457% | 512% | 136% | 211% |
| CAGR | 26.56% | 28.22% | 12.52% | 16.86% |
| Volatility (ann) | 21.77% | 23.96% | 16.76% | 19.54% |
| Sharpe Ratio | 1.01 | 1.00 | 0.55 | 0.70 |
| Sortino Ratio | 1.27 | 1.28 | 0.79 | 0.86 |
| **Calmar Ratio** | **0.76** | **0.80** | 0.63 | 0.50 |
| Max Drawdown | -34.79% | -35.06% | -19.89% | -33.72% |
| VaR 95% (daily) | 1.97% | 2.15% | 1.71% | 1.75% |
| CVaR 95% (daily) | 3.20% | 3.49% | 2.40% | 2.94% |
| Alpha (ann) | 9.09% | 10.68% | 1.20% | — |
| Beta | 0.97 | 1.00 | 0.59 | — |
| Info Ratio | 0.79 | 0.73 | -0.30 | — |

### How to read these results

**Momentum and Composite** are high-beta strategies (Beta ~1.0). They do not protect in crashes — Max Drawdown reaches ~35%, comparable to SPY. Their edge is entirely in Alpha: 9–10% annualised excess return after adjusting for market exposure, with Sharpe above 1.0 out-of-sample. For an investor willing to hold through a -35% drawdown, the long-run compounding significantly outperforms buy-and-hold.

**Risk-Managed** is a different product: Beta 0.59, Max Drawdown -19.89%, Calmar 0.63 vs 0.50 for SPY. The trade-off is a CAGR of 12.52% OOS, below SPY's 16.86%. This strategy makes sense for an investor who cannot or will not hold through a -35% drawdown.

**On the high OOS CAGR of Composite (28.22%):** the IS→OOS increase is unusual and requires honest interpretation. It is largely driven by Fold 11 (2024 H1), where NVDA returned ~+80% and the momentum signal correctly held it with significant weight. This is not look-ahead bias, but it is concentrated event risk. The median OOS Sharpe of 1.14 is the more representative long-run expectation.

---

## Walk-Forward Validation

**Setup:** 3-year rolling training window, 6-month OOS evaluation period, 15 folds (2016–2026).

| Strategy | Median Sharpe | Median CAGR | % Positive Folds | Worst Fold | Best Fold |
|---|---|---|---|---|---|
| Momentum | 1.27 | 26.2% | 93% | -1.05 | 3.29 |
| Composite | 1.14 | 26.0% | 87% | -0.97 | 3.69 |
| Risk-Managed | 0.53 | 12.2% | 80% | -2.29 | 3.29 |

Momentum is the most consistent strategy: 93% of folds with positive Sharpe and the highest median. The Composite generates slightly more Alpha but at lower consistency. Risk-Managed has the weakest tail: its worst fold (-2.29 in 2022 H1) is worse than the other two despite its defensive design, because the overlays reduce average returns enough to make bad folds more painful in relative terms.

**Fold 11 (2024 H1) — Sharpe 3.29/3.69:** NVDA returned ~+80% driven by the AI infrastructure boom. The momentum signal correctly identified it as top-ranked entering the period. Not look-ahead bias, but concentrated event risk — not the repeatable edge.

**Fold 7 (2022 H1) — worst fold for all strategies:** the Fed's hawkish pivot triggered a growth→value rotation. Momentum strategies held high-momentum growth stocks that became the hardest hit. This is the canonical momentum crash and should be expected to recur in future rate regime changes.

---

## Risk & Cost Assumptions

| Parameter | Value | Notes |
|---|---|---|
| Commission | 0.10% per trade | Conservative estimate for retail on Interactive Brokers large-caps |
| Slippage | 0.05% per trade | Realistic for liquid US large-caps (>$500M ADV) |
| Execution | Next open after signal | Avoids look-ahead bias on close prices |
| Rebalancing | Month-end | Risk overlays applied daily (Risk-Managed only) |
| Max leverage | 2.0x | Hard cap in vol scaling (`clip(0.5, 2.0)`) |

Transaction costs are simulated as `turnover × 2 × commission_rate`. At median monthly one-way turnover of ~19% (Momentum) and ~30% (Composite), annual cost drag is approximately 45–70 bps — material but not dominant relative to the Alpha generated.

---

## Limitations & Known Issues

- **No live execution.** The notebook produces target weights; it does not interface with any broker API. Converting to a live system requires an execution layer (e.g., Alpaca or IBKR API) and a daily scheduler.
- **EOD data only.** All signals are computed on end-of-day prices. Intraday dynamics, opening gaps, and overnight risk are not modelled.
- **No explicit sector constraints.** In periods of strong factor concentration (e.g., AI/tech in 2024), the portfolio can become an implicit sector bet. A sector exposure breakdown over time is a planned addition.
- **Turnover bug (Composite strategy):** max one-way turnover exceeds 100% in some months, indicating an accounting issue in the turnover calculation. Under investigation.
- **Momentum crash risk.** All three strategies have meaningful exposure to momentum factor reversals (see Fold 7). There is no explicit protection against rapid regime changes.

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
```bash
python src/strategy.py    # prints last weights for all three strategies
python src/walkforward.py # runs full walk-forward validation
```

**Dependencies:** `pandas`, `numpy`, `yfinance`, `scipy`, `matplotlib`, `seaborn`, `lightgbm`, `arch`