# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Hierarchical Risk Parity (HRP) portfolio construction.

Implements the Lopéz de Prado (2016) HRP algorithm, which groups assets
into a hierarchy of correlation clusters before allocating risk. Unlike
flat risk parity, HRP prevents "risk leakage" between correlated blocks
(e.g., FX pairs and equity indices sharing a 2026 global-liquidity factor).

Algorithm:
1. **Distance matrix**: d_ij = sqrt((1 - ρ_ij) / 2) — metric distance.
2. **Hierarchical clustering**: Ward linkage on the distance matrix.
3. **Quasi-diagonalisation**: Reorder assets so similar assets are adjacent.
4. **Recursive bisection**: Allocate risk top-down; within each cluster,
   weight by inverse variance. Clusters weighted by their total variance.

HRP advantages over plain risk parity:
- No matrix inversion required (numerically stable for large N).
- Naturally handles block-correlated universes (FX + Equity + Commodity).
- Out-of-sample Sharpe consistently higher than Markowitz (Lopéz de Prado 2016).

Typical usage::

    from systematic_macro.portfolio.hrp import HRPOptimizer
    hrp = HRPOptimizer()
    weights = hrp.compute_weights(returns)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree  # type: ignore[import]
from scipy.spatial.distance import squareform  # type: ignore[import]
from loguru import logger

from systematic_macro.utils.covariance import CovarianceEstimator


@dataclass
class HRPOptimizer:
    """Hierarchical Risk Parity portfolio weight calculator.

    Attributes:
        cov_method: Covariance estimator method (``"ledoit_wolf"`` recommended).
        linkage_method: Hierarchical clustering linkage
            (``"ward"``, ``"single"``, ``"complete"``, ``"average"``).
        signal_tilt: If > 0, tilts HRP weights toward signal direction.
            Value is the blending weight on signal vs pure HRP (0–1).
        min_obs: Minimum observations for covariance estimation.
        max_position: Maximum absolute weight per asset.
    """

    cov_method: Literal["ledoit_wolf", "oas", "sample"] = "ledoit_wolf"
    linkage_method: str = "ward"
    signal_tilt: float = 0.3
    min_obs: int = 30
    max_position: float = 0.25

    def __post_init__(self) -> None:
        valid_linkage = {"ward", "single", "complete", "average"}
        if self.linkage_method not in valid_linkage:
            raise ValueError(
                f"linkage_method must be one of {valid_linkage}, "
                f"got {self.linkage_method!r}"
            )
        if not (0.0 <= self.signal_tilt <= 1.0):
            raise ValueError(f"signal_tilt must be in [0,1], got {self.signal_tilt}")
        if not (0 < self.max_position <= 1.0):
            raise ValueError(f"max_position must be in (0,1], got {self.max_position}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        returns: pd.DataFrame,
        signal: pd.Series | None = None,
    ) -> pd.Series:
        """Compute HRP weights for a universe of assets.

        Args:
            returns: (T × N) returns DataFrame (NaN rows dropped).
            signal: Optional (N,) signal z-scores for signal-tilted HRP.
                Must share the same index as ``returns.columns``.

        Returns:
            (N,) Series of portfolio weights summing to 1.0 in absolute value.

        Raises:
            ValueError: If fewer than 2 assets or insufficient observations.
        """
        if returns.shape[1] < 2:
            raise ValueError(f"Need ≥ 2 assets, got {returns.shape[1]}.")

        estimator = CovarianceEstimator(
            method=self.cov_method, annualise=False, min_obs=self.min_obs
        )
        cov = estimator.fit(returns)
        corr = estimator.correlation_matrix(returns)

        # Step 1: Distance matrix
        dist = self._corr_to_distance(corr)

        # Step 2: Hierarchical clustering
        ordered_tickers = self._quasi_diagonalise(dist)
        logger.debug("HRP cluster order: {}", ordered_tickers)

        # Step 3: Recursive bisection
        cov_ordered = cov.loc[ordered_tickers, ordered_tickers]
        weights_raw = self._recursive_bisection(cov_ordered, ordered_tickers)

        weights = pd.Series(weights_raw, index=ordered_tickers).reindex(
            returns.columns
        )

        if signal is not None and self.signal_tilt > 0:
            weights = self._apply_signal_tilt(weights, signal)

        weights = weights.clip(-self.max_position, self.max_position)
        # Renormalise to sum-of-abs = 1
        total = weights.abs().sum()
        if total > 0:
            weights = weights / total

        logger.info(
            "HRP: n_assets={} | eff_N={:.1f} | max_w={:.3f}",
            len(weights),
            1.0 / (weights**2).sum() if (weights**2).sum() > 0 else 0,
            float(weights.abs().max()),
        )
        return weights

    def compute_rolling_weights(
        self,
        returns: pd.DataFrame,
        window: int = 126,
        step: int = 21,
        signal: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute HRP weights on a rolling basis.

        Args:
            returns: (T × N) returns DataFrame.
            window: Lookback window for covariance estimation.
            step: Rebalancing step in periods.
            signal: Optional (T × N) signal panel for signal tilt.

        Returns:
            (T × N) weight DataFrame (forward-filled between rebalance dates).

        Raises:
            ValueError: If ``window`` < ``min_obs``.
        """
        if window < self.min_obs:
            raise ValueError(
                f"window={window} < min_obs={self.min_obs}."
            )

        weight_rows: dict[pd.Timestamp, pd.Series] = {}
        dates = returns.index

        for i in range(window, len(dates), step):
            date = dates[i]
            hist = returns.iloc[i - window : i]
            sig_row = signal.loc[date] if signal is not None and date in signal.index else None

            try:
                w = self.compute_weights(hist, signal=sig_row)
                weight_rows[date] = w
            except ValueError as exc:
                logger.warning("HRP skipping {}: {}", date, exc)
                continue

        if not weight_rows:
            return pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

        weight_df = pd.DataFrame(weight_rows).T
        return weight_df.reindex(returns.index).ffill().fillna(0.0)

    # ------------------------------------------------------------------
    # Internal — HRP algorithm steps
    # ------------------------------------------------------------------

    @staticmethod
    def _corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
        """Convert correlation matrix to correlation distance matrix.

        Uses d_ij = sqrt((1 - ρ_ij) / 2), which satisfies triangle inequality.

        Args:
            corr: (N × N) correlation DataFrame.

        Returns:
            (N × N) distance DataFrame ∈ [0, 1].
        """
        dist = np.sqrt((1.0 - corr.to_numpy(dtype=float)) / 2.0)
        np.fill_diagonal(dist, 0.0)
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)

    def _quasi_diagonalise(self, dist: pd.DataFrame) -> list[str]:
        """Reorder assets via hierarchical clustering (quasi-diagonalisation).

        Args:
            dist: (N × N) distance DataFrame.

        Returns:
            Ordered list of asset tickers.
        """
        tickers = list(dist.index)
        dist_np = dist.to_numpy(dtype=float)

        # squareform requires condensed form
        condensed = squareform(dist_np, checks=False)
        link = linkage(condensed, method=self.linkage_method)

        # Extract leaf order from dendrogram
        root, _ = to_tree(link, rd=True)
        ordered_idx = self._get_leaf_order(root)
        return [tickers[i] for i in ordered_idx]

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        tickers: list[str],
    ) -> dict[str, float]:
        """Allocate weights top-down by recursive bisection.

        At each split, allocate the cluster's budget proportionally to
        inverse variance of each sub-cluster.

        Args:
            cov: (N × N) ordered covariance DataFrame.
            tickers: Ordered asset list.

        Returns:
            Dict of {ticker: weight}.
        """
        weights = {t: 1.0 for t in tickers}
        clusters = [tickers]

        while clusters:
            new_clusters: list[list[str]] = []
            for cluster in clusters:
                if len(cluster) == 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = self._cluster_var(cov, left)
                var_right = self._cluster_var(cov, right)

                total_var = var_left + var_right
                if total_var == 0:
                    alpha = 0.5
                else:
                    # Inverse-variance allocation
                    alpha = 1.0 - var_left / total_var

                for t in left:
                    weights[t] *= alpha
                for t in right:
                    weights[t] *= 1.0 - alpha

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return weights

    @staticmethod
    def _cluster_var(cov: pd.DataFrame, tickers: list[str]) -> float:
        """Compute the variance of an equal-weight portfolio of ``tickers``.

        Args:
            cov: Full covariance matrix.
            tickers: Subset of tickers in this cluster.

        Returns:
            Portfolio variance of equal-weight cluster.
        """
        sub = cov.loc[tickers, tickers].to_numpy(dtype=float)
        n = len(tickers)
        w = np.ones(n) / n
        return float(w @ sub @ w)

    def _apply_signal_tilt(
        self, hrp_weights: pd.Series, signal: pd.Series
    ) -> pd.Series:
        """Blend HRP weights with signal direction.

        Blended weight = (1 - tilt) × |HRP| × sign(signal) + tilt × signal_w
        where signal_w is signal z-score normalised to unit gross leverage.

        Args:
            hrp_weights: (N,) pure HRP weights.
            signal: (N,) z-score signal per asset.

        Returns:
            (N,) tilted weights.
        """
        aligned = signal.reindex(hrp_weights.index).fillna(0.0)
        sig_norm = aligned / (aligned.abs().sum() + 1e-12)

        # Apply signal direction to HRP magnitudes, then blend
        hrp_signed = hrp_weights.abs() * np.sign(aligned.where(aligned != 0, hrp_weights))
        blended = (1.0 - self.signal_tilt) * hrp_signed + self.signal_tilt * sig_norm
        return blended

    @staticmethod
    def _get_leaf_order(node: object) -> list[int]:
        """Recursively extract leaf order from a scipy ClusterNode.

        Args:
            node: Root :class:`scipy.cluster.hierarchy.ClusterNode`.

        Returns:
            Ordered list of leaf indices.
        """
        if node.is_leaf():  # type: ignore[union-attr]
            return [node.id]  # type: ignore[union-attr]
        return (
            HRPOptimizer._get_leaf_order(node.left)  # type: ignore[union-attr]
            + HRPOptimizer._get_leaf_order(node.right)  # type: ignore[union-attr]
        )
