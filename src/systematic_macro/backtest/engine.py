# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Walk-forward backtesting engine with IC gating and production monitoring.

Implements the full five-stage pipeline gate logic:
- Stage 3: IS/OOS walk-forward with ICIR, net Sharpe, and MDD gates.
- Stage 4: Marginal Sharpe and correlation screens.
- Stage 5: Rolling 60-day IC monitoring vs IS IC baseline.

Typical usage::

    from systematic_macro.backtest.engine import WalkForwardEngine
    engine = WalkForwardEngine(is_years=5, oos_years=1, step_months=6)
    result = engine.run(signal, returns)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
from loguru import logger

from systematic_macro.utils.metrics import (
    bonferroni_sharpe_threshold,
    compute_icir,
    compute_marginal_sharpe,
    compute_max_drawdown,
    compute_rolling_ic,
    compute_sharpe,
    compute_net_ic,
)


class FoldResult(NamedTuple):
    """Results for a single walk-forward fold.

    Attributes:
        fold_id: Zero-based fold index.
        is_start: In-sample start date.
        is_end: In-sample end date.
        oos_start: Out-of-sample start date.
        oos_end: Out-of-sample end date.
        is_icir: In-sample ICIR.
        oos_icir: Out-of-sample ICIR.
        oos_sharpe: Net OOS Sharpe ratio.
        oos_mdd: OOS maximum drawdown.
        passed_gate: Whether this fold passed all Stage 3 gates.
    """

    fold_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_icir: float
    oos_icir: float
    oos_sharpe: float
    oos_mdd: float
    passed_gate: bool


@dataclass
class BacktestResult:
    """Aggregated walk-forward backtest results.

    Attributes:
        folds: List of per-fold results.
        oos_returns: Concatenated OOS portfolio returns.
        oos_ic: Concatenated OOS IC series.
        is_ic_baseline: Mean IS ICIR across all passed folds.
        n_tests: Number of signal variants evaluated (for Bonferroni).
    """

    folds: list[FoldResult] = field(default_factory=list)
    oos_returns: pd.Series = field(default_factory=pd.Series)
    oos_ic: pd.Series = field(default_factory=pd.Series)
    is_ic_baseline: float = 0.0
    n_tests: int = 1

    def summary(self) -> dict[str, object]:
        """Return a summary dict of key performance metrics.

        Returns:
            Dict with keys: n_folds, n_passed, pass_rate, oos_sharpe,
            oos_mdd, oos_icir, is_icir_mean, is_icir_std.
        """
        passed = [f for f in self.folds if f.passed_gate]
        return {
            "n_folds": len(self.folds),
            "n_passed": len(passed),
            "pass_rate": len(passed) / max(len(self.folds), 1),
            "oos_sharpe": compute_sharpe(self.oos_returns),
            "oos_mdd": compute_max_drawdown(self.oos_returns),
            "oos_icir": compute_icir(self.oos_ic),
            "is_icir_mean": float(np.mean([f.is_icir for f in self.folds])),
            "is_icir_std": float(np.std([f.is_icir for f in self.folds], ddof=1)),
        }

    def passes_portfolio_gate(
        self,
        portfolio_returns: pd.Series,
        delta_sharpe_threshold: float = 0.05,
        correlation_threshold: float = 0.6,
    ) -> bool:
        """Stage 4 gate: marginal Sharpe and correlation screen.

        Args:
            portfolio_returns: Existing portfolio period returns.
            delta_sharpe_threshold: Minimum ΔSharpe required.
            correlation_threshold: Maximum correlation with portfolio.

        Returns:
            ``True`` if signal passes both Stage 4 gates.
        """
        delta = compute_marginal_sharpe(portfolio_returns, self.oos_returns)
        corr = self.oos_returns.corr(portfolio_returns)

        gate_sharpe = delta >= delta_sharpe_threshold
        gate_corr = abs(corr) < correlation_threshold

        logger.info(
            "Stage 4 gate: ΔSharpe={:.3f} (≥{:.2f}={}) | corr={:.2f} (<{:.1f}={})",
            delta,
            delta_sharpe_threshold,
            gate_sharpe,
            corr,
            correlation_threshold,
            gate_corr,
        )
        return gate_sharpe or gate_corr


@dataclass
class WalkForwardEngine:
    """Walk-forward backtesting engine implementing Stage 3 of the pipeline.

    Attributes:
        is_years: In-sample window length in years.
        oos_years: Out-of-sample window length in years.
        step_months: Rolling step size in months.
        icir_threshold: Minimum OOS ICIR gate (Stage 3).
        sharpe_threshold: Minimum OOS net Sharpe gate (Stage 3).
        mdd_vol_multiplier: MDD gate: MDD < multiplier × monthly vol.
        tc_bps: Transaction cost in basis points for net Sharpe/IC.
        turnover_estimate: Estimated one-way portfolio turnover per period.
        n_tests: Number of signal variants (for Bonferroni correction).
        ic_method: ``"spearman"`` or ``"pearson"``.
    """

    is_years: int = 5
    oos_years: int = 1
    step_months: int = 6
    icir_threshold: float = 0.5
    sharpe_threshold: float = 0.5
    mdd_vol_multiplier: float = 2.0
    tc_bps: float = 5.0
    turnover_estimate: float = 0.1
    n_tests: int = 1
    ic_method: str = "spearman"

    def __post_init__(self) -> None:
        """Validate and apply Bonferroni correction to Sharpe threshold."""
        if self.is_years < 1:
            raise ValueError(f"is_years must be ≥ 1, got {self.is_years}")
        if self.oos_years < 1:
            raise ValueError(f"oos_years must be ≥ 1, got {self.oos_years}")
        if self.step_months < 1:
            raise ValueError(f"step_months must be ≥ 1, got {self.step_months}")
        if self.tc_bps < 0:
            raise ValueError(f"tc_bps must be ≥ 0, got {self.tc_bps}")

        adjusted = bonferroni_sharpe_threshold(self.n_tests, self.sharpe_threshold)
        if adjusted != self.sharpe_threshold:
            logger.info(
                "Bonferroni-adjusted Sharpe threshold: {:.3f} → {:.3f} ({} tests)",
                self.sharpe_threshold,
                adjusted,
                self.n_tests,
            )
            self.sharpe_threshold = adjusted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
        portfolio_weights: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Execute walk-forward backtest across all folds.

        Args:
            signal: (T × N) signal z-scores.
            returns: (T × N) period returns (same columns and index).
            portfolio_weights: (T × N) optional portfolio weights used to
                compute portfolio returns from the signal.

        Returns:
            :class:`BacktestResult` with fold-level and aggregate statistics.

        Raises:
            ValueError: If ``signal`` and ``returns`` don't have matching
                columns or insufficient history.
        """
        if not signal.columns.equals(returns.columns):
            raise ValueError("signal and returns must have identical columns.")

        folds = self._generate_folds(signal.index)
        if not folds:
            raise ValueError(
                f"Insufficient history for {self.is_years}yr IS + "
                f"{self.oos_years}yr OOS. Need ≥ "
                f"{(self.is_years + self.oos_years) * 252} trading days."
            )

        fold_results: list[FoldResult] = []
        all_oos_returns: list[pd.Series] = []
        all_oos_ic: list[pd.Series] = []

        for fold_id, (is_start, is_end, oos_start, oos_end) in enumerate(folds):
            logger.info(
                "Fold {}/{}: IS {:%Y-%m-%d}→{:%Y-%m-%d} | OOS {:%Y-%m-%d}→{:%Y-%m-%d}",
                fold_id + 1,
                len(folds),
                is_start,
                is_end,
                oos_start,
                oos_end,
            )

            is_sig = signal.loc[is_start:is_end]
            is_ret = returns.loc[is_start:is_end]
            oos_sig = signal.loc[oos_start:oos_end]
            oos_ret = returns.loc[oos_start:oos_end]

            is_ic = compute_rolling_ic(is_sig, is_ret.shift(-1), self.ic_method)
            oos_ic = compute_rolling_ic(oos_sig, oos_ret.shift(-1), self.ic_method)

            is_icir = compute_icir(is_ic)
            oos_icir = compute_icir(oos_ic)

            if portfolio_weights is not None:
                oos_wts = portfolio_weights.loc[oos_start:oos_end]
                oos_port_ret = (oos_wts * oos_ret).sum(axis=1)
            else:
                # Signal-proportional weights (simplified)
                oos_port_ret = oos_sig.mul(oos_ret).mean(axis=1)

            gross_ic_mean = float(oos_ic.mean()) if not oos_ic.empty else 0.0
            net_ic_val = compute_net_ic(
                gross_ic_mean,
                self.turnover_estimate,
                self.tc_bps,
                signal_vol=max(float(oos_port_ret.std() * np.sqrt(252)), 0.01),
            )
            oos_net_ret = oos_port_ret - (self.tc_bps / 10_000) * self.turnover_estimate
            oos_sharpe = compute_sharpe(oos_net_ret)
            oos_mdd = compute_max_drawdown(oos_net_ret)
            monthly_vol = float(oos_net_ret.std() * np.sqrt(21))
            mdd_gate = oos_mdd < self.mdd_vol_multiplier * monthly_vol

            passed = (
                oos_icir >= self.icir_threshold
                and oos_sharpe >= self.sharpe_threshold
                and mdd_gate
            )

            logger.info(
                "  IS ICIR={:.3f} | OOS ICIR={:.3f} (≥{:.1f}={}) | "
                "OOS Sharpe={:.3f} (≥{:.2f}={}) | MDD={:.3f} (gate={})",
                is_icir,
                oos_icir,
                self.icir_threshold,
                oos_icir >= self.icir_threshold,
                oos_sharpe,
                self.sharpe_threshold,
                oos_sharpe >= self.sharpe_threshold,
                oos_mdd,
                mdd_gate,
            )

            fold_results.append(
                FoldResult(
                    fold_id=fold_id,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    is_icir=is_icir,
                    oos_icir=oos_icir,
                    oos_sharpe=oos_sharpe,
                    oos_mdd=oos_mdd,
                    passed_gate=passed,
                )
            )
            all_oos_returns.append(oos_net_ret)
            all_oos_ic.append(oos_ic)

        combined_returns = pd.concat(all_oos_returns).sort_index() if all_oos_returns else pd.Series(dtype=float)
        combined_ic = pd.concat(all_oos_ic).sort_index() if all_oos_ic else pd.Series(dtype=float)
        passed_folds = [f for f in fold_results if f.passed_gate]
        is_ic_baseline = float(np.mean([f.is_icir for f in passed_folds])) if passed_folds else 0.0

        return BacktestResult(
            folds=fold_results,
            oos_returns=combined_returns,
            oos_ic=combined_ic,
            is_ic_baseline=is_ic_baseline,
            n_tests=self.n_tests,
        )

    def monitor_live(
        self,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
        is_ic_baseline: float,
        window: int = 60,
        ratio_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Stage 5: Rolling IC monitoring vs IS baseline.

        Flags dates where the rolling IC / IS-baseline ratio drops below
        ``ratio_threshold`` for two consecutive periods.

        Args:
            signal: Live signal (T × N).
            returns: Live returns (T × N).
            is_ic_baseline: Mean IS ICIR from the backtest phase.
            window: Rolling IC monitoring window (days).
            ratio_threshold: IC ratio trigger (0.5 = flag if IC < 50% of IS).

        Returns:
            DataFrame with columns: ``rolling_ic``, ``ic_ratio``, ``flagged``.

        Raises:
            ValueError: If ``is_ic_baseline`` is 0.
        """
        if is_ic_baseline == 0.0:
            raise ValueError("is_ic_baseline must be non-zero for ratio comparison.")

        fwd = returns.shift(-1)
        rolling_ic = compute_rolling_ic(signal, fwd, self.ic_method)

        monitor = pd.DataFrame({
            "rolling_ic": rolling_ic,
            "ic_ratio": rolling_ic.abs() / abs(is_ic_baseline),
        })
        monitor["below_threshold"] = monitor["ic_ratio"] < ratio_threshold
        monitor["flagged"] = (
            monitor["below_threshold"]
            & monitor["below_threshold"].shift(1).fillna(False)
        )
        n_flags = int(monitor["flagged"].sum())
        if n_flags > 0:
            logger.warning(
                "Stage 5 monitor: {} flag(s) — IC ratio below {:.0%} for "
                "2+ consecutive periods. Review signal.",
                n_flags,
                ratio_threshold,
            )
        return monitor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_folds(
        self,
        index: pd.DatetimeIndex,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate IS/OOS date tuples for walk-forward.

        Args:
            index: Full DatetimeIndex of available data.

        Returns:
            List of (is_start, is_end, oos_start, oos_end) tuples.
        """
        is_days = self.is_years * 252
        oos_days = self.oos_years * 252
        step_days = int(self.step_months * 21)

        min_required = is_days + oos_days
        if len(index) < min_required:
            return []

        folds = []
        cursor = 0

        while cursor + min_required <= len(index):
            is_start = index[cursor]
            is_end = index[min(cursor + is_days - 1, len(index) - 1)]
            oos_start_idx = cursor + is_days
            oos_end_idx = min(oos_start_idx + oos_days - 1, len(index) - 1)

            if oos_start_idx >= len(index):
                break

            oos_start = index[oos_start_idx]
            oos_end = index[oos_end_idx]
            folds.append((is_start, is_end, oos_start, oos_end))
            cursor += step_days

        return folds
