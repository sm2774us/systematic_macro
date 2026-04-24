# Solution Explanation

## Design Philosophy

### Why these three signals in 2026?

**Carry** remains the most durable risk premium in macro markets. In 2026, the carry landscape is particularly rich: the Fed is cutting while the BoJ is hiking (first time since 2007), creating the largest G10 rate differential dispersion in a decade. Commodity futures curves are in persistent backwardation across energy and metals due to supply-chain fragmentation, delivering strong roll yield. We vol-adjust carry to prevent high-vol assets from dominating the cross-section.

**Momentum** is powered by geopolitical structural trends that persist over 3–12 month horizons: energy supply nationalism, AI capex cycle concentration in US equities, and EM currency fragmentation. We use multi-horizon TSMOM+XSMOM and add a 200-day MA regime filter to avoid momentum in trending-down markets (where it turns to reversal).

**Flow/Positioning** is the highest-frequency information asymmetry signal. In 2026, COT data shows record crowding in AI/tech equity longs and extreme JPY shorts (pre-BoJ pivot). Contrarian positioning at z-score extremes (>1.5σ) has historically delivered the strongest IC — we flip the signal only at those extremes.

---

## Engineering Decisions

### Library-first, notebook-second
`systematic_macro` is a proper installable package (`poetry install` runs `pip install -e .`). The notebook imports it like any third-party library — no `sys.path` hacks, no relative imports.

### Walk-forward discipline
- **IS=5yr, OOS=1yr, step=6mo** gives ~3–6 OOS folds per signal on 10yr history.
- We use **net IC** (after TC drag) in gate logic — not gross IC. Most published backtests fail here.
- **Bonferroni correction** scales the Sharpe gate by `√(log(n_tests))` — the Harvey-Liu-Zhu (2016) haircut for financial factor fishing.

### Statistical caveats we encode in code
- ICIR gate of 0.5 with 12 OOS IC observations has SE ≈ 0.29 — marginal statistical power. We flag this in the `monitor_live` output.
- ΔSharpe of 0.05 is below the estimation noise floor. The `passes_portfolio_gate` method therefore applies an OR logic: pass if ΔSharpe ≥ threshold **or** correlation < threshold.

### Covariance estimation
Portfolio optimiser uses rolling 63-day (1 quarter) sample covariance. This is intentionally simple — shrinkage estimators (Ledoit-Wolf) would improve out-of-sample performance but add a dependency. The `regularisation` parameter in `PortfolioOptimizer` adds L2 ridge to the MV problem as a practical alternative.

### Risk Parity implementation
Uses the Maillard et al. (2010) iterative ERC algorithm — O(N²) per step, converges in <50 iterations for N≤30. Tilted by signal direction so positions align with alpha view while equalising risk contribution.

---

## Testing Strategy

- **100% line coverage** enforced via `--cov-fail-under=100` in pytest config.
- Fixtures are session-scoped and use deterministic seeds — tests are fully reproducible.
- Edge cases covered: zero std, insufficient observations, mismatched shapes, invalid parameters.
- No mocking of core numerical logic — all tests run actual computations on synthetic data.

---

## 2026 Trade Ideas Encoded in Universe Definitions

| Asset Class | Long | Short | Signal Driver |
|-------------|------|-------|---------------|
| G10 FX | JPY (carry reversal) | EUR | Flow: record JPY short unwind + BoJ hike |
| Commodities | Gold, Oil front | NG next | Carry: backwardation + geopolitical premium |
| Equities | India, US (SPY) | Germany (EWG) | Momentum: AI capex + EU deindustrialisation |
| Rates | Short 30Y UST | Long 10Y | Carry: curve steepener as Fed cuts short end |

These are implemented as universe defaults in `data/fetcher.py` and demonstrated in the notebook.
