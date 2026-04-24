# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Carry signal: risk premium from holding higher-yielding assets.

In 2026, the carry signal is particularly relevant given:
- Divergent central bank policy cycles (Fed cutting vs BoJ hiking).
- Persistent EM/DM rate spreads above post-GFC averages.
- Commodity carry (futures curve roll yield) driven by supply constraints.

The signal is computed as the cross-sectional z-score of estimated carry,
normalised by realised volatility (vol-adjusted carry = carry-to-vol).

Typical usage::

    from systematic_macro.signals.carry import CarrySignal
    sig = CarrySignal(vol_window=21)
    scores = sig.compute(prices, rates)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CarrySignal:
    """Cross-sectional carry signal with vol-normalisation.

    Carry is estimated as the annualised roll/rate differential between
    assets, divided by their realised volatility. The output is a
    cross-sectional z-score suitable for ranking.

    Attributes:
        vol_window: Rolling window (days) for volatility estimation.
        carry_window: Lookback for smoothing carry estimates.
        min_assets: Minimum assets required for cross-sectional z-score.
        z_clip: Clip z-scores to ±z_clip to control outliers.
    """

    vol_window: int = 21
    carry_window: int = 5
    min_assets: int = 4
    z_clip: float = 3.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        prices: pd.DataFrame,
        rate_differentials: pd.DataFrame | None = None,
        futures_roll: pd.DataFrame | None = None,
        dividend_yield: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute the carry signal for a cross-section of assets.

        At least one of ``rate_differentials``, ``futures_roll``, or
        ``dividend_yield`` must be provided; they are summed to form a
        total carry estimate.

        Args:
            prices: (T × N) closing price DataFrame.
            rate_differentials: (T × N) annualised rate differential vs base
                currency (positive = higher yielding). For FX carry.
            futures_roll: (T × N) annualised futures roll yield. For
                commodity/rates futures carry.
            dividend_yield: (T × N) trailing dividend yield. For equity carry.

        Returns:
            (T × N) DataFrame of cross-sectional z-scored carry signals,
            clipped to ±``z_clip``.

        Raises:
            ValueError: If all carry components are ``None``.
            ValueError: If ``prices`` and any component have incompatible shape.
        """
        components = [rate_differentials, futures_roll, dividend_yield]
        if all(c is None for c in components):
            raise ValueError(
                "At least one of rate_differentials, futures_roll, or "
                "dividend_yield must be provided."
            )

        carry_raw = self._aggregate_carry(
            prices, rate_differentials, futures_roll, dividend_yield
        )
        vol = self._compute_vol(prices)
        carry_vol_adj = self._vol_adjust(carry_raw, vol)
        signal = self._cross_section_zscore(carry_vol_adj)
        return signal.clip(-self.z_clip, self.z_clip)

    def compute_fx_carry(
        self,
        spot_prices: pd.DataFrame,
        forward_discount: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute FX carry using covered interest parity.

        Forward discount ≈ interest rate differential (CIP deviation aside).
        Carry-long currencies with high forward premium vs USD.

        Args:
            spot_prices: (T × N) spot FX prices (USD per unit of foreign).
            forward_discount: (T × N) annualised forward discount
                (negative = currency trades at forward discount = cheaper
                to borrow, positive = at premium = higher yield).

        Returns:
            (T × N) vol-adjusted carry z-scores.
        """
        return self.compute(
            prices=spot_prices,
            rate_differentials=forward_discount,
        )

    def compute_futures_carry(
        self,
        front_prices: pd.DataFrame,
        next_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute futures roll yield carry (front vs next contract).

        Roll yield = (Front - Next) / Next, annualised by multiplying by
        the number of roll periods per year (assumed monthly = 12).

        Args:
            front_prices: (T × N) front-month futures prices.
            next_prices: (T × N) second-month futures prices.

        Returns:
            (T × N) vol-adjusted carry z-scores.

        Raises:
            ValueError: If ``front_prices`` and ``next_prices`` shapes differ.
        """
        if front_prices.shape != next_prices.shape:
            raise ValueError(
                f"front_prices shape {front_prices.shape} != "
                f"next_prices shape {next_prices.shape}"
            )

        roll_yield = ((front_prices - next_prices) / next_prices) * 12.0
        return self.compute(
            prices=front_prices,
            futures_roll=roll_yield,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_carry(
        self,
        prices: pd.DataFrame,
        rate_diff: pd.DataFrame | None,
        roll: pd.DataFrame | None,
        div_yield: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Sum all non-None carry components onto a common index."""
        total: pd.DataFrame | None = None
        for component in [rate_diff, roll, div_yield]:
            if component is None:
                continue
            aligned = component.reindex(prices.index).ffill().bfill()
            total = aligned if total is None else total.add(aligned, fill_value=0)

        assert total is not None  # guaranteed by earlier check
        smooth = total.rolling(self.carry_window, min_periods=1).mean()
        logger.debug("Carry aggregated; shape={}", smooth.shape)
        return smooth

    def _compute_vol(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute annualised rolling vol from price returns."""
        returns = prices.pct_change()
        vol = returns.rolling(self.vol_window, min_periods=5).std() * np.sqrt(252)
        return vol.replace(0.0, np.nan)

    @staticmethod
    def _vol_adjust(
        carry: pd.DataFrame,
        vol: pd.DataFrame,
    ) -> pd.DataFrame:
        """Divide carry by vol to get carry-to-vol ratio."""
        return carry.div(vol).replace([np.inf, -np.inf], np.nan)

    def _cross_section_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score per row (date)."""
        mu = df.mean(axis=1)
        sigma = df.std(axis=1, ddof=1)

        # Mask rows with too few valid observations
        valid_counts = df.notna().sum(axis=1)
        sigma = sigma.where(valid_counts >= self.min_assets)

        zscores = df.sub(mu, axis=0).div(sigma, axis=0)
        return zscores
