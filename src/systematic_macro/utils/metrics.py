# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Statistical and performance metrics for systematic signal research.

Provides vectorised implementations of IC, ICIR, Sharpe, MDD, and TCA
metrics used throughout the pipeline's gate logic and monitoring.

Typical usage::

    from systematic_macro.utils.metrics import compute_icir, compute_sharpe
    icir = compute_icir(ic_series)
    sharpe = compute_sharpe(returns, freq=252)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


def compute_ic(
    signal: pd.Series,
    forward_return: pd.Series,
    method: str = "spearman",
) -> float:
    """Compute the Information Coefficient between a signal and forward return.

    Args:
        signal: Cross-sectional signal values at time t.
        forward_return: Realised returns at time t+h for the same assets.
        method: Correlation method; ``"spearman"`` (rank IC) or ``"pearson"``.

    Returns:
        Scalar IC value in [-1, 1].

    Raises:
        ValueError: If ``method`` is not ``"spearman"`` or ``"pearson"``.
        ValueError: If ``signal`` and ``forward_return`` have fewer than 2
            overlapping observations.
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    aligned = pd.concat([signal, forward_return], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError("Need ≥2 overlapping observations to compute IC.")

    s = aligned.iloc[:, 0].to_numpy(dtype=float)
    r = aligned.iloc[:, 1].to_numpy(dtype=float)

    if method == "spearman":
        ic, _ = stats.spearmanr(s, r)
    else:
        ic, _ = stats.pearsonr(s, r)

    return float(ic)


def compute_rolling_ic(
    signal: pd.DataFrame,
    forward_return: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Compute period-by-period IC across a panel of assets.

    For each date row in *signal*, computes cross-sectional IC against the
    same row in *forward_return*. Both DataFrames must share dates (index)
    and assets (columns).

    Args:
        signal: (T × N) DataFrame of signal values.
        forward_return: (T × N) DataFrame of forward returns.
        method: ``"spearman"`` or ``"pearson"``.

    Returns:
        pd.Series of IC values indexed by date.
    """
    ic_values: dict[pd.Timestamp, float] = {}
    common_dates = signal.index.intersection(forward_return.index)

    for date in common_dates:
        s_row = signal.loc[date].dropna()
        r_row = forward_return.loc[date].reindex(s_row.index).dropna()
        s_row = s_row.reindex(r_row.index)

        if len(r_row) < 4:
            logger.debug(f"Skipping {date}: only {len(r_row)} valid observations.")
            continue
        ic_values[date] = compute_ic(s_row, r_row, method=method)

    return pd.Series(ic_values, name="IC")


def compute_icir(ic_series: pd.Series, min_obs: int = 12) -> float:
    """Compute the IC Information Ratio (mean IC / std IC).

    Args:
        ic_series: Time series of period IC values.
        min_obs: Minimum number of IC observations required.

    Returns:
        ICIR scalar. Returns ``0.0`` when ``ic_series`` has fewer than
        ``min_obs`` non-null values or std is zero.

    Raises:
        ValueError: If ``min_obs`` < 2.
    """
    if min_obs < 2:
        raise ValueError(f"min_obs must be ≥ 2, got {min_obs}")

    clean = ic_series.dropna()
    if len(clean) < min_obs:
        logger.warning(
            f"ICIR: only {len(clean)} obs (need {min_obs}); returning 0.0"
        )
        return 0.0

    mu = float(clean.mean())
    sigma = float(clean.std(ddof=1))
    if np.isclose(sigma, 0.0, atol=1e-12):
        return 0.0

    return mu / sigma


def compute_sharpe(
    returns: pd.Series,
    freq: int = 252,
    risk_free: float = 0.0,
) -> float:
    """Compute the annualised Sharpe ratio.

    Args:
        returns: Period returns (not log). Can be daily, weekly, monthly.
        freq: Number of periods per year (252=daily, 52=weekly, 12=monthly).
        risk_free: Annualised risk-free rate; will be scaled to per-period.

    Returns:
        Annualised Sharpe ratio. Returns ``0.0`` if std is zero.

    Raises:
        ValueError: If ``freq`` ≤ 0.
    """
    if freq <= 0:
        raise ValueError(f"freq must be > 0, got {freq}")

    clean = returns.dropna()
    if clean.empty:
        return 0.0

    rf_period = risk_free / freq
    excess = clean - rf_period
    mu = float(excess.mean())
    sigma = float(excess.std(ddof=1))

    if sigma == 0.0:
        return 0.0

    return float(mu / sigma * np.sqrt(freq))


def compute_max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from a series of period returns.

    Args:
        returns: Period returns (not log, not prices).

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.25 = 25% drawdown).
        Returns ``0.0`` for empty or all-zero series.
    """
    clean = returns.dropna()
    if clean.empty:
        return 0.0

    cum = (1.0 + clean).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return float(-drawdown.min())


def compute_net_ic(
    gross_ic: float,
    turnover: float,
    tc_bps: float = 5.0,
    signal_vol: float = 1.0,
) -> float:
    """Estimate net IC after transaction cost drag.

    Approximation: TC drag on IC ≈ (turnover × tc_bps / 10_000) / signal_vol

    Args:
        gross_ic: Gross IC before costs.
        turnover: Portfolio one-way turnover per period as a fraction (0–1).
        tc_bps: One-way transaction cost in basis points.
        signal_vol: Annualised signal return volatility (used as denominator
            when normalising the cost drag).

    Returns:
        Estimated net IC (may be negative).

    Raises:
        ValueError: If ``tc_bps`` < 0 or ``turnover`` not in [0, 1].
    """
    if tc_bps < 0:
        raise ValueError(f"tc_bps must be ≥ 0, got {tc_bps}")
    if not (0.0 <= turnover <= 1.0):
        raise ValueError(f"turnover must be in [0, 1], got {turnover}")
    if signal_vol <= 0.0:
        raise ValueError(f"signal_vol must be > 0, got {signal_vol}")

    drag = (turnover * tc_bps / 10_000.0) / signal_vol
    return gross_ic - drag


def compute_marginal_sharpe(
    portfolio_returns: pd.Series,
    signal_returns: pd.Series,
    weight: float = 0.05,
    freq: int = 252,
) -> float:
    """Compute the marginal Sharpe improvement from adding a signal.

    Blends *signal_returns* into *portfolio_returns* at *weight* and measures
    the Sharpe improvement.

    Args:
        portfolio_returns: Existing portfolio period returns.
        signal_returns: Candidate signal period returns.
        weight: Allocation fraction for the new signal (0–1).
        freq: Annualisation factor.

    Returns:
        ΔSharpe = Sharpe(blended) − Sharpe(portfolio).

    Raises:
        ValueError: If ``weight`` not in (0, 1).
    """
    if not (0.0 < weight < 1.0):
        raise ValueError(f"weight must be in (0, 1), got {weight}")

    aligned = pd.concat(
        [portfolio_returns, signal_returns], axis=1
    ).dropna()
    p = aligned.iloc[:, 0]
    s = aligned.iloc[:, 1]

    blended = (1.0 - weight) * p + weight * s
    return compute_sharpe(blended, freq=freq) - compute_sharpe(p, freq=freq)


def compute_calmar(returns: pd.Series, freq: int = 252) -> float:
    """Compute the Calmar ratio (annualised return / max drawdown).

    Args:
        returns: Period returns.
        freq: Annualisation factor.

    Returns:
        Calmar ratio, or ``0.0`` if MDD is zero.
    """
    ann_ret = float((1.0 + returns.dropna()).prod() ** (freq / max(len(returns), 1)) - 1)
    mdd = compute_max_drawdown(returns)
    return ann_ret / mdd if mdd > 0 else 0.0


def bonferroni_sharpe_threshold(n_tests: int, base_threshold: float = 0.5) -> float:
    """Apply a Bonferroni-style haircut to the Sharpe gate for multiple testing.

    Based on Harvey, Liu & Zhu (2016): the minimum t-stat for significance
    scales roughly with sqrt(log(n_tests)).

    Args:
        n_tests: Number of signal variants tested.
        base_threshold: Baseline Sharpe threshold (pre-correction).

    Returns:
        Adjusted minimum Sharpe threshold.

    Raises:
        ValueError: If ``n_tests`` < 1.
    """
    if n_tests < 1:
        raise ValueError(f"n_tests must be ≥ 1, got {n_tests}")
    if n_tests == 1:
        return base_threshold

    correction = float(np.sqrt(np.log(n_tests)))
    return base_threshold * correction
