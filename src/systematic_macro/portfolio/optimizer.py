# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Portfolio optimisation: signal combination and position sizing.

Implements three portfolio construction approaches:
1. **Risk Parity**: Equal risk contribution across signals/assets.
2. **Mean-Variance**: Marginal Sharpe optimisation with regularisation.
3. **Signal-Scaled**: Direct vol-scaled position sizing from z-scores.

All methods return portfolio weights that sum to 1 in absolute value
(long-short portfolio normalisation).

Typical usage::

    from systematic_macro.portfolio.optimizer import PortfolioOptimizer
    opt = PortfolioOptimizer(method="risk_parity")
    weights = opt.compute_weights(signals, returns_cov)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class PortfolioOptimizer:
    """Multi-method portfolio weight optimiser for signal-based strategies.

    Attributes:
        method: Optimisation approach: ``"signal_scaled"``, ``"risk_parity"``,
            or ``"mean_variance"``.
        target_vol: Target annualised portfolio volatility (e.g. 0.10 = 10%).
        max_position: Maximum absolute weight per asset (e.g. 0.20 = 20%).
        min_weight: Minimum absolute weight to include (smaller set to 0).
        regularisation: L2 regularisation for mean-variance (lambda).
        signal_weights: Dict mapping signal name to blend weight.
            Defaults to equal weighting.
    """

    method: Literal["signal_scaled", "risk_parity", "mean_variance"] = "signal_scaled"
    target_vol: float = 0.10
    max_position: float = 0.25
    min_weight: float = 0.01
    regularisation: float = 0.1
    signal_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_methods = {"signal_scaled", "risk_parity", "mean_variance"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        if not (0 < self.target_vol < 1):
            raise ValueError(f"target_vol must be in (0, 1), got {self.target_vol}")
        if not (0 < self.max_position <= 1):
            raise ValueError(f"max_position must be in (0, 1], got {self.max_position}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
        cov_window: int = 63,
    ) -> pd.DataFrame:
        """Compute portfolio weights for each date.

        Args:
            signal: (T × N) composite signal z-scores.
            returns: (T × N) historical period returns for covariance.
            cov_window: Rolling window for covariance estimation.

        Returns:
            (T × N) DataFrame of portfolio weights. Rows with insufficient
            data return zeros.

        Raises:
            ValueError: If ``signal`` and ``returns`` columns don't match.
        """
        if not set(signal.columns).issubset(returns.columns):
            raise ValueError("signal columns must be a subset of returns columns.")

        ret_aligned = returns[signal.columns]
        weights_list: list[pd.Series] = []

        for date in signal.index:
            row_signal = signal.loc[date].dropna()
            if len(row_signal) < 2:
                weights_list.append(pd.Series(0.0, index=signal.columns))
                continue

            hist = ret_aligned.loc[:date].iloc[-(cov_window + 1):-1]
            if len(hist) < 10:
                weights_list.append(pd.Series(0.0, index=signal.columns))
                continue

            cov = hist[row_signal.index].cov().to_numpy(dtype=float)

            if self.method == "signal_scaled":
                w = self._signal_scaled(row_signal, cov)
            elif self.method == "risk_parity":
                w = self._risk_parity(row_signal, cov)
            else:
                w = self._mean_variance(row_signal, cov)

            w_series = pd.Series(w, index=row_signal.index)
            w_full = w_series.reindex(signal.columns).fillna(0.0)
            weights_list.append(w_full)

        weights = pd.DataFrame(weights_list, index=signal.index)
        return self._apply_position_limits(weights)

    def blend_signals(
        self,
        signals: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Combine multiple signal DataFrames into a single composite signal.

        Uses ``signal_weights`` if provided, otherwise equal-weighted.

        Args:
            signals: Dict of signal name → (T × N) z-score DataFrame.

        Returns:
            (T × N) composite z-score DataFrame.

        Raises:
            ValueError: If ``signals`` is empty.
            ValueError: If provided ``signal_weights`` keys don't match.
        """
        if not signals:
            raise ValueError("signals dict must not be empty.")

        names = list(signals.keys())

        if self.signal_weights:
            missing = set(names) - set(self.signal_weights.keys())
            if missing:
                raise ValueError(
                    f"signal_weights missing keys: {missing}"
                )
            weights = {k: self.signal_weights[k] for k in names}
        else:
            weights = {k: 1.0 / len(names) for k in names}

        # Normalise weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        composite: pd.DataFrame | None = None
        for name, df in signals.items():
            w = weights[name]
            weighted = df * w
            composite = weighted if composite is None else composite.add(weighted, fill_value=0.0)

        assert composite is not None
        logger.info(
            "Blended {} signals with weights {}",
            len(signals),
            {k: f"{v:.2f}" for k, v in weights.items()},
        )
        return composite

    def vol_scale_weights(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        vol_window: int = 21,
    ) -> pd.DataFrame:
        """Scale portfolio weights to hit ``target_vol``.

        Computes realised portfolio vol and scales the whole weight vector
        so the expected portfolio vol equals ``target_vol``.

        Args:
            weights: (T × N) raw weights.
            returns: (T × N) period returns.
            vol_window: Lookback for portfolio vol estimation.

        Returns:
            (T × N) vol-scaled weights.
        """
        aligned = returns[weights.columns]
        port_ret = (weights * aligned.shift(1)).sum(axis=1)
        port_vol = (
            port_ret.rolling(vol_window, min_periods=5)
            .std()
            .mul(np.sqrt(252))
            .replace(0.0, np.nan)
        )

        scale = (self.target_vol / port_vol).clip(0.1, 5.0).fillna(1.0)
        scaled = weights.mul(scale, axis=0)
        return self._apply_position_limits(scaled)

    # ------------------------------------------------------------------
    # Internal optimisation methods
    # ------------------------------------------------------------------

    def _signal_scaled(
        self,
        signal: pd.Series,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Weight proportional to signal, scaled by inverse volatility."""
        vols = np.sqrt(np.diag(cov))
        vols = np.where(vols == 0, np.nan, vols)
        raw = signal.to_numpy(dtype=float) / vols
        raw = np.nan_to_num(raw, nan=0.0)
        norm = np.sum(np.abs(raw))
        return raw / norm if norm > 0 else raw

    def _risk_parity(
        self,
        signal: pd.Series,
        cov: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """Equal risk contribution weights, tilted by signal direction.

        Uses the Maillard et al. (2010) iterative ERC algorithm.

        Args:
            signal: Cross-sectional signal z-scores.
            cov: (N × N) covariance matrix.
            max_iter: Maximum Newton iterations.
            tol: Convergence tolerance.

        Returns:
            (N,) weight array, normalised to sum-of-abs = 1.
        """
        n = len(signal)
        w = np.ones(n) / n
        directions = np.sign(signal.to_numpy(dtype=float))
        directions = np.where(directions == 0, 1.0, directions)

        for _ in range(max_iter):
            port_var = float(w @ cov @ w)
            if port_var <= 0:
                break
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk / port_var
            target_contrib = np.ones(n) / n

            grad = marginal_risk - (target_contrib / (w + 1e-10)) * port_var
            step = 0.01
            w_new = w - step * grad
            w_new = np.abs(w_new) * directions
            w_new /= np.sum(np.abs(w_new)) + 1e-12

            if np.max(np.abs(w_new - w)) < tol:
                break
            w = w_new

        return w / (np.sum(np.abs(w)) + 1e-12)

    def _mean_variance(
        self,
        signal: pd.Series,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Regularised mean-variance weights (signal as expected return).

        Args:
            signal: Cross-sectional z-scores used as expected returns.
            cov: (N × N) covariance matrix.

        Returns:
            (N,) weight array, normalised.
        """
        mu = signal.to_numpy(dtype=float)
        reg_cov = cov + self.regularisation * np.eye(len(mu))

        try:
            w = np.linalg.solve(reg_cov, mu)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance; falling back to signal_scaled.")
            return self._signal_scaled(signal, cov)

        norm = np.sum(np.abs(w))
        return w / norm if norm > 0 else np.zeros_like(w)

    def _apply_position_limits(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Clip weights to ±max_position and zero out weights below min_weight."""
        clipped = weights.clip(-self.max_position, self.max_position)
        below_min = clipped.abs() < self.min_weight
        clipped[below_min] = 0.0
        return clipped
