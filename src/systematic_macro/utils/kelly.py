# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Kelly Criterion position sizing for systematic macro strategies.

The Kelly Criterion computes the theoretically optimal fraction of capital
to allocate to a strategy to maximise long-run geometric growth. In practice,
**fractional Kelly** (typically 25–50%) is used to limit drawdown risk and
account for parameter estimation error.

For a multi-asset systematic book, the **multivariate Kelly** solution uses
the precision matrix (Σ⁻¹) to compute the full optimal weight vector that
accounts for asset covariances — preventing doubling down on hidden common
factors (e.g., the 2026 global liquidity factor driving both JPY and credit).

For a $100M book with a 0.4 IC signal on EURUSD:
- Estimated alpha = IC × vol ≈ 0.4 × 8% = 3.2% per month
- Kelly fraction = α / σ² ≈ 0.032 / 0.0064 ≈ 5× (gross!) → use 25% Kelly → 1.25×
- Dollar sizing = 1.25 × $100M = $125M notional (reasonable for liquid FX)

Typical usage::

    from systematic_macro.utils.kelly import KellySizer
    sizer = KellySizer(fraction=0.25, max_leverage=2.0)
    weights = sizer.compute(expected_returns, cov_matrix)
    dollar_sizes = sizer.dollar_size(weights, book_size=100_000_000)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class KellySizer:
    """Fractional Kelly position sizer using covariance-aware allocation.

    Attributes:
        fraction: Kelly fraction f ∈ (0, 1]. Full Kelly = 1.0, half = 0.5.
            Use 0.25 for systematic macro (high estimation error environment).
        max_leverage: Maximum gross leverage cap (e.g. 2.0 = 200% gross).
        max_position: Maximum single-asset weight (e.g. 0.20 = 20% of book).
        min_edge: Minimum expected return needed to allocate (filters noise).
        use_precision: If True, uses Σ⁻¹ (multivariate Kelly) for better
            hedging. If False, uses independent per-asset Kelly.
    """

    fraction: float = 0.25
    max_leverage: float = 2.0
    max_position: float = 0.20
    min_edge: float = 0.0
    use_precision: bool = True

    def __post_init__(self) -> None:
        if not (0 < self.fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if self.max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {self.max_leverage}")
        if not (0 < self.max_position <= 1.0):
            raise ValueError(f"max_position must be in (0, 1], got {self.max_position}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> pd.Series:
        """Compute fractional Kelly weights for a cross-section of assets.

        Full multivariate Kelly: w* = f × Σ⁻¹ × μ
        Per-asset Kelly fallback: w_i* = f × μ_i / σ_i²

        Args:
            expected_returns: (N,) Series of expected period returns per asset.
                Can be raw IC × vol estimates or direct return forecasts.
            cov_matrix: (N × N) covariance DataFrame (same index as returns).

        Returns:
            (N,) Series of portfolio weights ∈ [-max_position, max_position],
            with total gross leverage ≤ max_leverage.

        Raises:
            ValueError: If ``expected_returns`` and ``cov_matrix`` are
                misaligned.
        """
        if not expected_returns.index.equals(cov_matrix.index):
            raise ValueError(
                "expected_returns index must match cov_matrix index."
            )

        # Filter assets with insufficient edge
        if self.min_edge > 0:
            mask = expected_returns.abs() >= self.min_edge
            expected_returns = expected_returns[mask]
            cov_matrix = cov_matrix.loc[mask, mask]

        if expected_returns.empty:
            logger.warning("Kelly: no assets pass min_edge filter; returning zeros.")
            return pd.Series(dtype=float)

        mu = expected_returns.to_numpy(dtype=float)
        cov_np = cov_matrix.to_numpy(dtype=float)

        raw_weights = self._kelly_weights(mu, cov_np)
        scaled = self.fraction * raw_weights

        # Apply position and leverage limits
        scaled = np.clip(scaled, -self.max_position, self.max_position)
        gross = float(np.sum(np.abs(scaled)))
        if gross > self.max_leverage:
            scaled = scaled * (self.max_leverage / gross)

        result = pd.Series(scaled, index=expected_returns.index, name="kelly_weight")
        logger.info(
            "Kelly: gross={:.2f}x | net={:.2f}x | n_assets={}",
            float(np.sum(np.abs(result))),
            float(np.sum(result)),
            len(result),
        )
        return result

    def compute_from_signal(
        self,
        signal: pd.Series,
        vol: pd.Series,
        cov_matrix: pd.DataFrame,
        ic_estimate: float = 0.05,
    ) -> pd.Series:
        """Compute Kelly weights from a cross-sectional signal.

        Converts z-score signal to expected returns using:
        μ_i = IC × σ_i × z_i  (Grinold's Fundamental Law)

        Args:
            signal: (N,) cross-sectional z-score signal.
            vol: (N,) annualised asset volatilities.
            cov_matrix: (N × N) covariance matrix.
            ic_estimate: Expected IC used to scale returns. Default 0.05.

        Returns:
            (N,) Kelly weights.

        Raises:
            ValueError: If ``ic_estimate`` is zero or negative.
        """
        if ic_estimate <= 0:
            raise ValueError(f"ic_estimate must be > 0, got {ic_estimate}")

        # Grinold's law: alpha_i = IC × sigma_i × z_i
        expected_returns = ic_estimate * vol * signal
        aligned = expected_returns.reindex(cov_matrix.index).dropna()
        return self.compute(aligned, cov_matrix.loc[aligned.index, aligned.index])

    def dollar_size(
        self,
        weights: pd.Series,
        book_size: float,
        asset_prices: pd.Series | None = None,
        lot_sizes: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Convert weight fractions to dollar notional and share/lot counts.

        Args:
            weights: (N,) Kelly weight fractions (−1 to 1).
            book_size: Total AUM in dollars (e.g. 100_000_000 = $100M).
            asset_prices: (N,) current prices for share/lot calculation.
                If None, only dollar notional is computed.
            lot_sizes: (N,) minimum lot sizes per asset (default 1).

        Returns:
            DataFrame with columns: ``weight``, ``notional_usd``,
            and optionally ``units`` (shares/contracts).

        Raises:
            ValueError: If ``book_size`` ≤ 0.
        """
        if book_size <= 0:
            raise ValueError(f"book_size must be > 0, got {book_size}")

        notional = weights * book_size
        result = pd.DataFrame({
            "weight": weights,
            "notional_usd": notional,
        })

        if asset_prices is not None:
            prices = asset_prices.reindex(weights.index)
            lots = lot_sizes.reindex(weights.index).fillna(1.0) if lot_sizes is not None else pd.Series(1.0, index=weights.index)
            raw_units = notional / prices
            result["units_raw"] = raw_units
            result["units_rounded"] = (raw_units / lots).apply(np.floor) * lots

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _kelly_weights(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute full or diagonal Kelly weights.

        Args:
            mu: (N,) expected return vector.
            cov: (N × N) covariance matrix.

        Returns:
            (N,) raw (pre-fraction, pre-clipping) Kelly weights.
        """
        if self.use_precision:
            try:
                reg_cov = cov + 1e-8 * np.eye(len(mu))
                prec = np.linalg.inv(reg_cov)
                w = prec @ mu
                return w
            except np.linalg.LinAlgError:
                logger.warning("Kelly: precision matrix inversion failed; using diagonal.")

        # Fallback: independent per-asset Kelly w_i = μ_i / σ_i²
        variances = np.diag(cov)
        variances = np.where(variances <= 0, np.nan, variances)
        return np.nan_to_num(mu / variances, nan=0.0)
