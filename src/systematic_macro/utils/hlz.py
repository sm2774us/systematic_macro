# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Harvey-Liu-Zhu (2016) multiple testing correction for strategy t-statistics.

The HLZ framework from "… and the Cross-Section of Expected Returns" (2016)
addresses the multiple-testing problem in backtested financial strategies.
As more strategies are tested, some will appear significant by chance alone.

Key outputs:
- ``min_tstat``: Minimum t-statistic required given N tests performed.
- ``hlz_haircut``: Multiplicative shrinkage applied to observed t-stats.
- ``adjusted_sharpe``: Net Sharpe after applying the HLZ haircut.

In a 2026 environment with high vol and crowded factor fishing, this
correction is critical — a Sharpe of 1.2 from a strategy tested 50 times
may be statistically indistinguishable from noise.

Typical usage::

    from systematic_macro.utils.hlz import HLZCorrection
    hlz = HLZCorrection(n_tests=50, sample_length=252*3)
    result = hlz.evaluate(observed_sharpe=1.2)
    print(result.adjusted_sharpe, result.passes_threshold)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy import stats
from loguru import logger


class HLZResult(NamedTuple):
    """Output of an HLZ multiple-testing evaluation.

    Attributes:
        observed_tstat: Raw t-statistic from the backtest.
        min_tstat: HLZ minimum t-statistic threshold.
        adjusted_tstat: Haircut-adjusted t-statistic.
        adjusted_sharpe: Adjusted annualised Sharpe ratio.
        haircut_pct: Percentage haircut applied (0–100).
        passes_threshold: Whether adjusted_tstat ≥ min_tstat.
    """

    observed_tstat: float
    min_tstat: float
    adjusted_tstat: float
    adjusted_sharpe: float
    haircut_pct: float
    passes_threshold: bool


@dataclass
class HLZCorrection:
    """Harvey-Liu-Zhu multiple-testing correction engine.

    Implements the HLZ (2016) framework for computing the minimum
    t-statistic required for a strategy to be considered significant
    given that ``n_tests`` strategies have been evaluated.

    Three haircut methods are provided:
    - ``"hlz"``: HLZ expected maximum statistic (default, most rigorous).
    - ``"bonferroni"``: Classical Bonferroni (most conservative).
    - ``"holm"``: Holm-Bonferroni step-down (intermediate).

    Attributes:
        n_tests: Total number of strategies/signals evaluated.
        sample_length: Number of return observations in the backtest.
        annual_factor: Annualisation factor (252 daily, 52 weekly, 12 monthly).
        significance: Target false-discovery rate (default 0.05 = 5%).
        correlation: Average pairwise correlation between test statistics.
            HLZ uses ρ=0.2 as their empirical estimate for factor strategies.
        method: Haircut methodology (``"hlz"``, ``"bonferroni"``, ``"holm"``).
    """

    n_tests: int = 1
    sample_length: int = 252
    annual_factor: int = 252
    significance: float = 0.05
    correlation: float = 0.2
    method: str = "hlz"

    def __post_init__(self) -> None:
        if self.n_tests < 1:
            raise ValueError(f"n_tests must be ≥ 1, got {self.n_tests}")
        if self.sample_length < 10:
            raise ValueError(f"sample_length must be ≥ 10, got {self.sample_length}")
        if not (0 < self.significance < 1):
            raise ValueError(f"significance must be in (0,1), got {self.significance}")
        if not (0.0 <= self.correlation < 1.0):
            raise ValueError(f"correlation must be in [0,1), got {self.correlation}")
        if self.method not in {"hlz", "bonferroni", "holm"}:
            raise ValueError(f"method must be 'hlz', 'bonferroni', or 'holm'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        observed_sharpe: float,
        freq: int | None = None,
    ) -> HLZResult:
        """Evaluate a single strategy's Sharpe against the HLZ threshold.

        Args:
            observed_sharpe: Annualised Sharpe ratio from the backtest.
            freq: Observations per year used for this strategy (overrides
                ``annual_factor`` if provided).

        Returns:
            :class:`HLZResult` with adjusted statistics and pass/fail flag.
        """
        af = freq if freq is not None else self.annual_factor
        t_obs = sharpe_to_tstat(observed_sharpe, self.sample_length, af)
        t_min = self.minimum_tstat()
        t_adj, haircut = self._apply_haircut(t_obs, t_min)
        sr_adj = tstat_to_sharpe(t_adj, self.sample_length, af)

        logger.info(
            "HLZ[{}] n_tests={} | t_obs={:.3f} | t_min={:.3f} | "
            "t_adj={:.3f} | SR_adj={:.3f} | haircut={:.1f}%",
            self.method, self.n_tests, t_obs, t_min, t_adj, sr_adj, haircut,
        )

        return HLZResult(
            observed_tstat=t_obs,
            min_tstat=t_min,
            adjusted_tstat=t_adj,
            adjusted_sharpe=sr_adj,
            haircut_pct=haircut,
            passes_threshold=t_adj >= t_min,
        )

    def minimum_tstat(self) -> float:
        """Compute the minimum t-statistic threshold for the chosen method.

        Returns:
            Minimum t-statistic a strategy must exceed (on its adjusted
            value) to be considered statistically significant.
        """
        if self.n_tests == 1:
            return float(stats.norm.ppf(1.0 - self.significance / 2.0))

        if self.method == "bonferroni":
            return float(
                stats.norm.ppf(1.0 - self.significance / (2.0 * self.n_tests))
            )

        if self.method == "holm":
            # Holm: same as Bonferroni for the first test in step-down
            return float(
                stats.norm.ppf(1.0 - self.significance / (2.0 * self.n_tests))
            )

        # HLZ expected maximum of correlated normals
        return self._hlz_expected_max()

    def batch_evaluate(
        self,
        sharpe_ratios: list[float],
        freq: int | None = None,
    ) -> list[HLZResult]:
        """Evaluate a batch of strategies, updating n_tests automatically.

        Args:
            sharpe_ratios: List of annualised Sharpe ratios.
            freq: Observations per year (overrides annual_factor).

        Returns:
            List of :class:`HLZResult` in the same order as input.
        """
        original_n = self.n_tests
        self.n_tests = len(sharpe_ratios)
        results = [self.evaluate(sr, freq=freq) for sr in sharpe_ratios]
        self.n_tests = original_n
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hlz_expected_max(self) -> float:
        """HLZ formula for expected maximum of N correlated normals.

        Based on Harvey, Liu & Zhu (2016) Eq. (3):
        E[max(Z_1,...,Z_N)] ≈ (1 - γ)Φ^{-1}(1-1/N) + γΦ^{-1}(1-1/(Ne))
        where γ is the Euler-Mascheroni constant.

        This is then adjusted for the average pairwise correlation ρ to
        account for the effective number of independent tests.
        """
        gamma = 0.5772156649  # Euler-Mascheroni constant
        n_eff = self._effective_tests()

        if n_eff <= 1:
            return float(stats.norm.ppf(1.0 - self.significance / 2.0))

        # Expected max of n_eff independent standard normals
        term1 = stats.norm.ppf(1.0 - 1.0 / n_eff)
        term2 = stats.norm.ppf(1.0 - 1.0 / (n_eff * math.e))
        e_max = (1.0 - gamma) * float(term1) + gamma * float(term2)

        # Scale for significance level
        z_alpha = float(stats.norm.ppf(1.0 - self.significance / 2.0))
        # Blend: weight expected-max by significance adjustment
        t_min = max(e_max, z_alpha)
        return float(t_min)

    def _effective_tests(self) -> float:
        """Compute effective number of independent tests given correlation ρ.

        Uses the Meff formula: Meff = 1 + (N-1)(1-ρ).
        For ρ=0 → Meff=N; for ρ=1 → Meff=1.
        """
        return 1.0 + (self.n_tests - 1) * (1.0 - self.correlation)

    def _apply_haircut(
        self, t_obs: float, t_min: float
    ) -> tuple[float, float]:
        """Apply proportional haircut: t_adj = t_obs × (t_min_single / t_min).

        The intuition: a single-test critical value (≈1.96 at 5%) is the
        "fair" hurdle. The ratio scales the observed stat proportionally.

        Args:
            t_obs: Observed t-statistic.
            t_min: Multiple-testing threshold.

        Returns:
            Tuple of (adjusted_tstat, haircut_percentage).
        """
        t_single = float(stats.norm.ppf(1.0 - self.significance / 2.0))
        if t_min <= 0:
            return t_obs, 0.0

        # Haircut factor: how much more demanding is the multiple-test threshold?
        haircut_factor = t_single / t_min  # < 1 when n_tests > 1
        t_adj = t_obs * haircut_factor
        haircut_pct = (1.0 - haircut_factor) * 100.0
        return float(t_adj), float(max(0.0, haircut_pct))


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def sharpe_to_tstat(
    sharpe: float, n_obs: int, annual_factor: int = 252
) -> float:
    """Convert annualised Sharpe ratio to a t-statistic.

    t = SR / sqrt(annual_factor) × sqrt(n_obs)

    Args:
        sharpe: Annualised Sharpe ratio.
        n_obs: Number of return observations.
        annual_factor: Periods per year.

    Returns:
        t-statistic.

    Raises:
        ValueError: If ``n_obs`` < 2 or ``annual_factor`` ≤ 0.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be ≥ 2, got {n_obs}")
    if annual_factor <= 0:
        raise ValueError(f"annual_factor must be > 0, got {annual_factor}")
    return sharpe / math.sqrt(annual_factor) * math.sqrt(n_obs)


def tstat_to_sharpe(
    tstat: float, n_obs: int, annual_factor: int = 252
) -> float:
    """Convert a t-statistic back to an annualised Sharpe ratio.

    SR = t / sqrt(n_obs) × sqrt(annual_factor)

    Args:
        tstat: t-statistic.
        n_obs: Number of return observations.
        annual_factor: Periods per year.

    Returns:
        Annualised Sharpe ratio.

    Raises:
        ValueError: If ``n_obs`` < 2 or ``annual_factor`` ≤ 0.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be ≥ 2, got {n_obs}")
    if annual_factor <= 0:
        raise ValueError(f"annual_factor must be > 0, got {annual_factor}")
    return tstat / math.sqrt(n_obs) * math.sqrt(annual_factor)
