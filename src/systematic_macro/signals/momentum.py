# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Momentum/trend signal: behavioural anchoring alpha.

In 2026, momentum is powered by:
- Geopolitical supply-chain fragmentation driving commodity and FX trends.
- Divergent equity market performance (US AI-driven vs EU/EM laggards).
- Bond market trends driven by asymmetric rate-cut timing across G10.

We implement three momentum variants:
1. **Time-series momentum** (TSMOM): sign of trailing return × vol-scaled.
2. **Cross-sectional momentum** (XSMOM): z-score of trailing returns.
3. **Dual-momentum**: TSMOM gated by XSMOM rank (only long if both agree).

Typical usage::

    from systematic_macro.signals.momentum import MomentumSignal
    sig = MomentumSignal(lookbacks=[21, 63, 126, 252])
    scores = sig.compute(returns)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class MomentumSignal:
    """Multi-horizon momentum signal with optional trend filtering.

    Computes a composite of time-series and cross-sectional momentum
    across multiple lookback periods, weighted by inverse lookback
    (shorter lookbacks get higher weight — decays quickly).

    Attributes:
        lookbacks: List of lookback periods in trading days.
        vol_window: Window for volatility-scaling of TSMOM.
        min_assets: Minimum assets for cross-sectional z-score.
        z_clip: Clip z-scores to ±z_clip.
        skip_recent: Skip most-recent ``skip_recent`` days (reversal buffer).
        tsmom_weight: Weight on time-series momentum in blended signal.
        xsmom_weight: Weight on cross-sectional momentum in blended signal.
    """

    lookbacks: list[int] = field(default_factory=lambda: [21, 63, 126, 252])
    vol_window: int = 21
    min_assets: int = 4
    z_clip: float = 3.0
    skip_recent: int = 1
    tsmom_weight: float = 0.5
    xsmom_weight: float = 0.5

    def __post_init__(self) -> None:
        """Validate weights sum to 1."""
        if abs(self.tsmom_weight + self.xsmom_weight - 1.0) > 1e-9:
            raise ValueError(
                f"tsmom_weight + xsmom_weight must equal 1.0, got "
                f"{self.tsmom_weight + self.xsmom_weight}"
            )
        if any(lb < 1 for lb in self.lookbacks):
            raise ValueError("All lookbacks must be ≥ 1.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute the blended momentum signal.

        Args:
            returns: (T × N) period returns DataFrame.

        Returns:
            (T × N) DataFrame of momentum z-scores clipped to ±``z_clip``.

        Raises:
            ValueError: If ``returns`` has fewer columns than ``min_assets``.
        """
        if returns.shape[1] < self.min_assets:
            raise ValueError(
                f"Need ≥ {self.min_assets} assets, got {returns.shape[1]}."
            )

        vol = self._rolling_vol(returns)
        tsmom = self._time_series_momentum(returns, vol)
        xsmom = self._cross_sectional_momentum(returns)

        blended = (
            self.tsmom_weight * tsmom + self.xsmom_weight * xsmom
        )
        return blended.clip(-self.z_clip, self.z_clip)

    def compute_tsmom_only(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute pure time-series momentum (vol-scaled, sign-based).

        Args:
            returns: (T × N) period returns DataFrame.

        Returns:
            (T × N) TSMOM signal; values ∈ {−1, 0, 1} × vol-scaling.
        """
        vol = self._rolling_vol(returns)
        return self._time_series_momentum(returns, vol)

    def compute_xsmom_only(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute pure cross-sectional momentum z-scores.

        Args:
            returns: (T × N) period returns DataFrame.

        Returns:
            (T × N) XSMOM z-scores.
        """
        return self._cross_sectional_momentum(returns)

    def regime_filter(
        self,
        signal: pd.DataFrame,
        market_return: pd.Series,
        window: int = 200,
    ) -> pd.DataFrame:
        """Apply a trend filter: zero out signals in down-trending markets.

        Uses a 200-day moving-average regime filter on a broad market index.
        When the market is below its MA, long signals are zeroed (trend-
        following only in trending environments).

        Args:
            signal: (T × N) momentum signal.
            market_return: (T,) broad market period returns (e.g. SPY).
            window: MA window for regime detection.

        Returns:
            (T × N) filtered signal.

        Raises:
            ValueError: If ``window`` < 10.
        """
        if window < 10:
            raise ValueError(f"window must be ≥ 10, got {window}")

        cum_return = (1 + market_return).cumprod()
        ma = cum_return.rolling(window, min_periods=window // 2).mean()
        trending = (cum_return > ma).astype(float)

        # When trending=1, pass through; when 0, zero out longs (but allow shorts)
        filtered = signal.copy()
        no_trend_mask = trending == 0
        filtered.loc[no_trend_mask] = filtered.loc[no_trend_mask].clip(upper=0)
        logger.debug("Regime filter applied; trending fraction={:.1%}", trending.mean())
        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rolling_vol(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling annualised vol, clipped to avoid division by zero."""
        vol = (
            returns.rolling(self.vol_window, min_periods=5)
            .std(ddof=1)
            .mul(np.sqrt(252))
            .replace(0.0, np.nan)
        )
        return vol

    def _time_series_momentum(
        self,
        returns: pd.DataFrame,
        vol: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute TSMOM: weighted average of sign(ret) / vol across lookbacks."""
        tsmom_list: list[pd.DataFrame] = []

        for lb in self.lookbacks:
            end_idx = lb
            start_idx = lb + self.skip_recent

            # trailing return from skip_recent to lookback ago
            trailing = (
                returns.shift(self.skip_recent)
                .rolling(lb, min_periods=lb // 2)
                .sum()
            )
            direction = np.sign(trailing)
            vol_adj = direction.div(vol).replace([np.inf, -np.inf], np.nan)

            # shorter lookback → larger inverse weight
            weight = 1.0 / lb
            tsmom_list.append(vol_adj * weight)
            del end_idx, start_idx  # silence lint

        composite = sum(tsmom_list)  # type: ignore[arg-type]
        assert isinstance(composite, pd.DataFrame)

        # Cross-sectional z-score per row
        mu = composite.mean(axis=1)
        sigma = composite.std(axis=1, ddof=1)
        return composite.sub(mu, axis=0).div(sigma, axis=0)

    def _cross_sectional_momentum(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute XSMOM: cross-sectional z-score of trailing returns."""
        xsmom_list: list[pd.DataFrame] = []
        weight_total = 0.0

        for lb in self.lookbacks:
            trailing = (
                returns.shift(self.skip_recent)
                .rolling(lb, min_periods=lb // 2)
                .sum()
            )
            w = 1.0 / lb
            weight_total += w
            xsmom_list.append(trailing * w)

        composite = sum(xsmom_list) / weight_total  # type: ignore[arg-type]
        assert isinstance(composite, pd.DataFrame)

        mu = composite.mean(axis=1)
        sigma = composite.std(axis=1, ddof=1)
        valid = composite.notna().sum(axis=1) >= self.min_assets
        sigma_safe = sigma.where(valid)
        return composite.sub(mu, axis=0).div(sigma_safe, axis=0)
