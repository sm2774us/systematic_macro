# Systematic Macro 2026
### Carry · Momentum · Flow — FX · Futures · Equities

[![CI](https://github.com/sm2774us/systematic_macro/actions/workflows/ci.yml/badge.svg)](https://github.com/sm2774us/systematic_macro/actions)
[![Coverage](https://codecov.io/gh/your-org/systematic-macro-2026/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/systematic-macro-2026)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![Tests](https://img.shields.io/badge/tests-287%20passing-brightgreen.svg)]()
[![Style](https://img.shields.io/badge/style-Google-blue.svg)](https://google.github.io/styleguide/pyguide.html)

An **institutional-grade**, end-to-end systematic macro alpha research pipeline implementing the three canonical alpha sources — **carry (risk premium)**, **momentum (behavioural anchoring)**, and **flow/positioning (information asymmetry)** — across G10 FX, liquid futures, and global equity ETFs.

Built for the **2026 macro and geopolitical environment**: divergent central bank cycles (Fed cutting / BoJ hiking), commodity supply fragmentation, AI-driven equity regime dispersion, and record JPY positioning unwinds.

Architected to the standards found at **Cubist, Citadel, and Two Sigma**: Ledoit-Wolf covariance shrinkage, Harvey-Liu-Zhu multiple-testing correction, Hurst/Kalman signal filtering, fractional Kelly via precision matrix, Hierarchical Risk Parity, and Polars LazyFrame Parquet pipelines.

---

## Architecture

```
systematic_macro/
├── data/
│   ├── fetcher.py          MarketDataFetcher — GBM synthetic + yfinance
│   └── parquet_loader.py   Polars LazyFrame Parquet (predicate + projection pushdown)
├── signals/
│   ├── carry.py            Vol-adjusted carry-to-vol cross-sectional z-score
│   ├── momentum.py         Multi-horizon TSMOM + XSMOM + 200d regime filter
│   └── flow.py             COT + OBV price-volume + options skew (contrarian at extremes)
├── portfolio/
│   ├── optimizer.py        Signal blending, signal-scaled / risk-parity / MV weights
│   └── hrp.py              Hierarchical Risk Parity (Lopez de Prado 2016)
└── utils/
    ├── metrics.py           IC, ICIR, Sharpe, MDD, net IC, marginal Sharpe
    ├── covariance.py        Ledoit-Wolf shrinkage, precision matrix, effective rank
    ├── hlz.py               Harvey-Liu-Zhu t-stat haircut (HLZ / Bonferroni / Holm)
    ├── hurst_kalman.py      Hurst R/S exponent + 1D Kalman filter (standard + adaptive)
    ├── kelly.py             Fractional multivariate Kelly (w*=f*Sigma_inv*mu)
    └── protocols.py         TradeableAsset Protocol — FXSpot, FuturesContract, EquityETF
```

---

## Quickstart

### Option 1 — Poetry (local, Python 3.13+)
```bash
git clone https://github.com/your-org/systematic-macro-2026.git
cd systematic-macro-2026
poetry install
poetry run pytest                          # 287 tests, 100% coverage
poetry run jupyter notebook notebooks/    # Full pipeline demo
```

### Option 2 — Docker (zero setup)
```bash
docker-compose up test        # Full test suite in Python 3.13-slim
docker-compose up notebook    # Jupyter on localhost:8888
```

---

## Signal Logic

| Signal | Mechanism | 2026 Macro Rationale |
|--------|-----------|----------------------|
| **Carry** | Vol-adj rate/roll/div yield z-score | Fed/BoJ divergence largest in decade; commodity backwardation from supply fragmentation |
| **Momentum** | Multi-horizon TSMOM+XSMOM (21/63/126/252d) + 200d regime filter | Geopolitical trend persistence; AI capex drives US-vs-EU equity regime divergence |
| **Flow** | COT net speculative + OBV + put/call skew (contrarian at >1.5σ) | Record JPY short unwind (BoJ pivot catalyst); extreme AI/tech equity crowding |

---

## Institutional-Grade Feature Set

### Signal Quality & Filtering
| Feature | Implementation | Purpose |
|---------|---------------|---------|
| **Hurst Exponent** | R/S analysis, log-log regression | Classify series — trending (H>0.55), mean-reverting (H<0.45), random walk (discard) |
| **Kalman Filter** | Standard + adaptive obs-noise variant | Optimal linear smoother; adapts gain as macro regime signal-to-noise ratio shifts |
| **HLZ Haircut** | Expected max of N correlated normals | Prevents strategies passing gates solely due to multiple-testing luck |

### Risk Estimation & Sizing
| Feature | Implementation | Purpose |
|---------|---------------|---------|
| **Ledoit-Wolf Shrinkage** | Analytical LW + OAS via sklearn | Invertible, conditioned covariance; prevents eigenvalue blow-up for N>10 assets |
| **Precision Matrix** | Regularised inversion of shrunken Sigma | True conditional asset dependencies; suppresses hidden factor double-counting |
| **Fractional Kelly** | w* = f * Sigma_inv * mu, Grinold's Law | Multivariate optimal sizing; dollar notional with lot rounding per asset class |
| **HRP** | Ward clustering, recursive bisection | No matrix inversion; handles FX/Equity/Commodity block correlation naturally |

### Infrastructure
| Feature | Implementation | Purpose |
|---------|---------------|---------|
| **Protocol typing** | `TradeableAsset` runtime Protocol | FXSpot (5dp, 100K lots), FuturesContract (point_value), EquityETF each sized correctly |
| **Polars LazyFrame** | `scan_parquet` with predicate/projection pushdown | 4-10x faster than pandas for 500-asset tick data; multi-core columnar I/O |

---

## Five-Stage Pipeline Gates

```
Stage 1 — Hypothesis:   Document why before touching data (carry/momentum/flow taxonomy)
Stage 2 — Data Hygiene: look-ahead · survivorship · point-in-time · >= 2 macro cycles
Stage 3 — Backtest:     ICIR > 0.5 OOS (HLZ-adjusted) | net Sharpe > 0.5 | MDD < 2x monthly vol
Stage 4 — Portfolio:    DeltaSharpe >= 0.05 OR corr < 0.6 | HRP replaces flat risk-parity
Stage 5 — Production:   60d rolling IC / IS-baseline < 0.5 for 2 months -> flag & review
```

---

## Test Coverage

287 tests across 14 modules — 100% line coverage enforced via `--cov-fail-under=100`.
All fixtures are session-scoped with deterministic seeds; zero mocking of numerical logic.

---

## License
MIT
