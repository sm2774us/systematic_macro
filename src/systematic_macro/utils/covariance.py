# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Ledoit-Wolf shrinkage covariance and precision matrix estimation.

In systematic macro, the sample covariance matrix is notoriously noisy for
N > 10 assets (eigenvalue blow-up, Marchenko-Pastur noise floor). The
**Ledoit-Wolf (2004)** analytical shrinkage estimator optimally blends the
sample covariance with a structured target (identity or constant-correlation)
to produce a well-conditioned, invertible covariance matrix.

The **precision matrix** (Σ⁻¹) encodes hedging relationships: entry (i,j)
of the precision matrix is non-zero only if assets i and j are conditionally
dependent after controlling for all other assets — the true "risk linkage".

In 2026, when JPY and EUR co-move due to a global liquidity squeeze, the
precision matrix will suppress the apparent diversification benefit while the
raw correlation matrix would miss the hidden factor.

Typical usage::

    from systematic_macro.utils.covariance import CovarianceEstimator
    est = CovarianceEstimator(method="ledoit_wolf")
    cov = est.fit(returns)
    prec = est.precision_matrix(returns)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS  # type: ignore[import]
from loguru import logger


@dataclass
class CovarianceEstimator:
    """Robust covariance matrix estimator with Ledoit-Wolf shrinkage.

    Supports three estimation methods:
    - ``"ledoit_wolf"``: Analytical LW shrinkage (default, O(N²T)).
    - ``"oas"``: Oracle Approximating Shrinkage (better for small T/N).
    - ``"sample"``: Classical sample covariance (no shrinkage; baseline).

    Attributes:
        method: Estimation method.
        annualise: If True, multiply by ``annual_factor`` (covariance units).
        annual_factor: Trading periods per year.
        min_obs: Minimum observations required relative to N assets.
    """

    method: Literal["ledoit_wolf", "oas", "sample"] = "ledoit_wolf"
    annualise: bool = True
    annual_factor: int = 252
    min_obs: int = 30

    def __post_init__(self) -> None:
        valid = {"ledoit_wolf", "oas", "sample"}
        if self.method not in valid:
            raise ValueError(f"method must be one of {valid}, got {self.method!r}")
        if self.annual_factor <= 0:
            raise ValueError(f"annual_factor must be > 0, got {self.annual_factor}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate the covariance matrix from a returns panel.

        Args:
            returns: (T × N) returns DataFrame. NaN rows are dropped.

        Returns:
            (N × N) covariance DataFrame (annualised if ``annualise=True``).

        Raises:
            ValueError: If fewer than ``min_obs`` clean rows remain.
            ValueError: If N > T (underdetermined; only for ``"sample"``).
        """
        clean = returns.dropna(how="any")
        n_obs, n_assets = clean.shape

        if n_obs < self.min_obs:
            raise ValueError(
                f"Need ≥ {self.min_obs} clean observations, got {n_obs}."
            )

        X = clean.to_numpy(dtype=float)
        cov_matrix = self._estimate_cov(X, n_obs, n_assets)

        if self.annualise:
            cov_matrix = cov_matrix * self.annual_factor

        return pd.DataFrame(
            cov_matrix,
            index=returns.columns,
            columns=returns.columns,
        )

    def precision_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute the precision matrix Σ⁻¹ using the shrunken covariance.

        The precision matrix encodes true conditional dependencies between
        assets. A non-zero (i,j) entry means assets i and j share a direct
        risk linkage not explained by other assets in the universe.

        Args:
            returns: (T × N) returns DataFrame.

        Returns:
            (N × N) precision matrix DataFrame.

        Raises:
            ValueError: If the covariance matrix is singular.
        """
        cov = self.fit(returns)
        cov_np = cov.to_numpy(dtype=float)

        try:
            # Use regularised inversion for numerical stability
            prec = np.linalg.inv(cov_np + 1e-10 * np.eye(len(cov_np)))
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Covariance matrix is singular even after regularisation. "
                "Reduce the asset universe or increase the lookback window."
            ) from exc

        logger.debug(
            "Precision matrix: condition number={:.1f}",
            float(np.linalg.cond(cov_np)),
        )
        return pd.DataFrame(
            prec, index=returns.columns, columns=returns.columns
        )

    def shrinkage_intensity(self, returns: pd.DataFrame) -> float:
        """Return the Ledoit-Wolf shrinkage intensity α ∈ [0, 1].

        α = 0: no shrinkage (pure sample covariance).
        α = 1: full shrinkage (pure structured target).

        Args:
            returns: (T × N) returns DataFrame.

        Returns:
            Shrinkage intensity. Returns 0.0 for ``"sample"`` method.

        Raises:
            ValueError: If fewer than ``min_obs`` observations.
        """
        if self.method == "sample":
            return 0.0

        clean = returns.dropna(how="any").to_numpy(dtype=float)
        if len(clean) < self.min_obs:
            raise ValueError(
                f"Need ≥ {self.min_obs} clean observations, got {len(clean)}."
            )

        estimator = LedoitWolf() if self.method == "ledoit_wolf" else OAS()
        estimator.fit(clean)
        alpha = float(getattr(estimator, "shrinkage_", 0.0))
        logger.debug("Ledoit-Wolf shrinkage intensity α={:.4f}", alpha)
        return alpha

    def correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return the shrinkage-based correlation matrix.

        Derived from the shrunken covariance: ρ_ij = σ_ij / (σ_i × σ_j).

        Args:
            returns: (T × N) returns DataFrame.

        Returns:
            (N × N) correlation matrix DataFrame ∈ [-1, 1].
        """
        cov = self.fit(returns)
        cov_np = cov.to_numpy(dtype=float)
        vols = np.sqrt(np.diag(cov_np))
        # Avoid division by zero
        vols = np.where(vols == 0, np.nan, vols)
        outer = np.outer(vols, vols)
        corr = cov_np / outer
        np.fill_diagonal(corr, 1.0)
        return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

    def effective_rank(self, returns: pd.DataFrame) -> float:
        """Compute effective rank of the covariance matrix.

        Effective rank = exp(entropy of normalised eigenvalues) — a measure
        of how many independent risk factors exist in the universe. Low
        effective rank in 2026 signals hidden factor concentration.

        Args:
            returns: (T × N) returns DataFrame.

        Returns:
            Effective rank ∈ [1, N].
        """
        cov = self.fit(returns).to_numpy(dtype=float)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        total = eigenvalues.sum()
        if total == 0:
            return 1.0
        p = eigenvalues / total
        entropy = float(-np.sum(p * np.log(p + 1e-15)))
        return float(np.exp(entropy))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _estimate_cov(
        self, X: np.ndarray, n_obs: int, n_assets: int
    ) -> np.ndarray:
        """Dispatch to the selected covariance estimator.

        Args:
            X: (T × N) float array of returns.
            n_obs: Number of observations.
            n_assets: Number of assets.

        Returns:
            (N × N) covariance matrix as ndarray.
        """
        if self.method == "sample":
            if n_obs <= n_assets:
                logger.warning(
                    "T={} ≤ N={}: sample covariance is rank-deficient.", n_obs, n_assets
                )
            return np.cov(X.T, ddof=1)

        estimator = LedoitWolf() if self.method == "ledoit_wolf" else OAS()
        estimator.fit(X)
        return estimator.covariance_
