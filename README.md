# Systematic Macro 2026
### Carry · Momentum · Flow — FX · Futures · Equities

[![CI](https://github.com/your-org/systematic-macro-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/systematic-macro-2026/actions)
[![Coverage](https://codecov.io/gh/your-org/systematic-macro-2026/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/systematic-macro-2026)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)

A production-grade, end-to-end systematic macro signal research pipeline implementing the three canonical alpha sources — **carry (risk premium)**, **momentum (behavioural anchoring)**, and **flow/positioning (information asymmetry)** — across G10 FX, liquid futures, and global equity ETFs.

Built specifically for the **2026 macro and geopolitical environment**: divergent central bank cycles, commodity supply fragmentation, AI-driven equity regime dispersion, and record JPY positioning unwinds.

---

## Architecture

```
systematic_macro/
├── data/           MarketDataFetcher — prices, returns, vol, universes
├── signals/
│   ├── carry.py    Vol-adjusted carry-to-vol cross-sectional z-score
│   ├── momentum.py Multi-horizon TSMOM + XSMOM blend with regime filter
│   └── flow.py     COT positioning + OBV flow + options skew (contrarian)
├── portfolio/      Signal blending, risk parity, mean-variance, vol-targeting
├── backtest/       Walk-forward engine: IS=5yr, OOS=1yr, step=6mo
└── utils/          IC, ICIR, Sharpe, MDD, net IC, Bonferroni correction
```

## Quickstart

### Option 1 — Poetry (local)
```bash
git clone https://github.com/your-org/systematic-macro-2026.git
cd systematic-macro-2026
poetry install
poetry run pytest                          # 100% coverage
poetry run jupyter notebook notebooks/
```

### Option 2 — Docker
```bash
docker-compose up test        # run full test suite
docker-compose up notebook    # launch Jupyter on :8888
```

## Signal Logic

| Signal | Source | 2026 Rationale |
|--------|--------|----------------|
| **Carry** | Rate differentials, futures roll yield, dividend yield | Fed/BoJ divergence; commodity backwardation (supply squeeze) |
| **Momentum** | Multi-horizon TSMOM + XSMOM (21/63/126/252d) | Geopolitical trend persistence; AI equity regime |
| **Flow** | COT net speculative + OBV + options skew (contrarian) | Crowded AI longs; extreme JPY short unwind |

## Pipeline Gates

```
Stage 3 (Backtest):   ICIR > 0.5 OOS | net Sharpe > 0.5 | MDD < 2× monthly vol
Stage 4 (Portfolio):  ΔSharpe ≥ 0.05 OR correlation < 0.6 with existing book
Stage 5 (Production): Flag if rolling 60d IC / IS-IC baseline < 0.5 for 2 months
```

Multiple-testing correction applied via Harvey-Liu-Zhu Bonferroni haircut.

## Project Structure

```
systematic-macro-2026/
├── src/systematic_macro/   # Installable library (poetry local install)
├── tests/                  # 100% pytest coverage
├── notebooks/              # GitHub-compatible JSON Jupyter notebook
├── pyproject.toml          # Poetry + tool config
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
└── SOLUTION_EXPLANATION.md
```

## License
MIT
