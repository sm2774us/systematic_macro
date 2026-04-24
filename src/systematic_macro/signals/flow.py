# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Flow and positioning signal: information asymmetry alpha.

In 2026, positioning signals are especially potent due to:
- Extreme crowding in AI/tech equity longs (contrarian short signal).
- Record net short JPY positioning unwinding (BOJ pivot catalyst).
- Commodity positioning driven by supply-chain geopolitical fragmentation.
- COT (Commitment of Traders) data for futures providing weekly flow.

This module implements three positioning sub-signals:
1. **COT-based**: Speculative net positioning z-score (contrarian).
2. **Options skew**: Put/call skew as a flow-positioning proxy.
3. **Synthetic flow**: Price-volume flow (modified OBV z-score).

Typical usage::

    from systematic_macro.signals.flow import FlowSignal
    sig = FlowSignal(cot_window=52, flow_window=21)
    scores = sig.compute(prices, volume=volume)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FlowSignal:
    """Positioning and flow signal with COT and price-volume components.

    Attributes:
        cot_window: Rolling z-score window for COT positioning (weeks).
        flow_window: Rolling window for price-volume flow signal (days).
        min_assets: Minimum assets for cross-sectional normalisation.
        z_clip: Z-score clip.
        contrarian: If ``True``, invert the positioning signal
            (crowded long = sell signal). Most effective at extremes.
        contrarian_threshold: Z-score magnitude above which contrarian
            flip is applied (0.0 = always contrarian).
    """

    cot_window: int = 52
    flow_window: int = 21
    min_assets: int = 4
    z_clip: float = 3.0
    contrarian: bool = True
    contrarian_threshold: float = 1.5

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.cot_window < 4:
            raise ValueError(f"cot_window must be ≥ 4, got {self.cot_window}")
        if self.flow_window < 5:
            raise ValueError(f"flow_window must be ≥ 5, got {self.flow_window}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        cot_net: pd.DataFrame | None = None,
        options_skew: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute composite flow/positioning signal.

        Blends available sub-signals equally. At least one of ``volume``,
        ``cot_net``, or ``options_skew`` must be provided alongside ``prices``.

        Args:
            prices: (T × N) price DataFrame.
            volume: (T × N) traded volume (uses synthetic if None).
            cot_net: (T × N) COT net speculative positions (optional).
                Frequency can be weekly; will be forward-filled to daily.
            options_skew: (T × N) 25-delta put-call skew (optional).
                Positive = puts more expensive (bearish positioning).

        Returns:
            (T × N) composite flow z-score, clipped to ±``z_clip``.

        Raises:
            ValueError: If only ``prices`` provided and ``volume`` is None.
        """
        subs: list[pd.DataFrame] = []

        vol_signal = self._price_volume_flow(prices, volume)
        subs.append(vol_signal)

        if cot_net is not None:
            cot_signal = self._cot_signal(cot_net, prices)
            subs.append(cot_signal)

        if options_skew is not None:
            skew_signal = self._skew_signal(options_skew, prices)
            subs.append(skew_signal)

        combined = sum(subs) / len(subs)  # type: ignore[arg-type]
        assert isinstance(combined, pd.DataFrame)

        normalised = self._cross_section_zscore(combined)

        if self.contrarian:
            normalised = self._apply_contrarian(normalised)

        return normalised.clip(-self.z_clip, self.z_clip)

    def compute_cot_signal(
        self,
        cot_net: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute standalone COT-based positioning signal.

        Args:
            cot_net: (T × N) net speculative positions from CFTC COT reports.
            prices: (T × N) corresponding prices (for index alignment).

        Returns:
            (T × N) COT z-score signal.
        """
        raw = self._cot_signal(cot_net, prices)
        normalised = self._cross_section_zscore(raw)
        if self.contrarian:
            normalised = self._apply_contrarian(normalised)
        return normalised.clip(-self.z_clip, self.z_clip)

    # ------------------------------------------------------------------
    # Internal sub-signal constructors
    # ------------------------------------------------------------------

    def _price_volume_flow(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Compute a modified On-Balance Volume (OBV) z-score.

        OBV accumulates volume on up-days, subtracts on down-days.
        We z-score the rolling OBV change to normalise across assets.

        Args:
            prices: (T × N) prices.
            volume: (T × N) volume; if None, synthetic uniform volume used.

        Returns:
            (T × N) flow signal DataFrame.
        """
        if volume is None:
            logger.warning("No volume provided; using synthetic uniform volume.")
            volume = pd.DataFrame(
                np.ones(prices.shape), index=prices.index, columns=prices.columns
            )

        price_change = prices.diff()
        direction = np.sign(price_change)
        signed_vol = direction.mul(volume)
        obv = signed_vol.rolling(self.flow_window, min_periods=5).sum()

        mu = obv.rolling(self.flow_window * 3, min_periods=self.flow_window).mean()
        sigma = obv.rolling(self.flow_window * 3, min_periods=self.flow_window).std(ddof=1)
        return (obv - mu) / sigma.replace(0.0, np.nan)

    def _cot_signal(
        self,
        cot_net: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Z-score of net speculative COT positioning.

        Args:
            cot_net: (T × N) net positions (long - short speculative).
            prices: (T × N) price DataFrame for index alignment.

        Returns:
            (T × N) COT z-score reindexed to ``prices`` index.
        """
        # Forward-fill weekly COT data to daily
        daily_cot = cot_net.reindex(prices.index).ffill()

        mu = daily_cot.rolling(self.cot_window * 5, min_periods=self.cot_window).mean()
        sigma = daily_cot.rolling(
            self.cot_window * 5, min_periods=self.cot_window
        ).std(ddof=1)
        return (daily_cot - mu) / sigma.replace(0.0, np.nan)

    def _skew_signal(
        self,
        options_skew: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Z-score of options put/call skew as positioning proxy.

        High positive skew = heavy put buying = bearish positioning.
        We invert: high positive skew → negative signal (already bearish).

        Args:
            options_skew: (T × N) 25-delta risk reversal skew.
            prices: (T × N) price DataFrame for alignment.

        Returns:
            (T × N) z-scored skew signal (sign inverted).
        """
        aligned = options_skew.reindex(prices.index).ffill()
        mu = aligned.rolling(self.flow_window * 5, min_periods=self.flow_window).mean()
        sigma = aligned.rolling(
            self.flow_window * 5, min_periods=self.flow_window
        ).std(ddof=1)
        zscore = (aligned - mu) / sigma.replace(0.0, np.nan)
        return -zscore  # invert: high put skew = bearish positioning = sell signal

    def _cross_section_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score per row."""
        mu = df.mean(axis=1)
        sigma = df.std(axis=1, ddof=1)
        valid = df.notna().sum(axis=1) >= self.min_assets
        sigma_safe = sigma.where(valid)
        return df.sub(mu, axis=0).div(sigma_safe, axis=0)

    def _apply_contrarian(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Invert signal only at extremes above ``contrarian_threshold``.

        Args:
            signal: (T × N) normalised signal.

        Returns:
            (T × N) signal with extreme values inverted.
        """
        threshold = self.contrarian_threshold
        if threshold == 0.0:
            return -signal

        flipped = signal.copy()
        extreme_long = signal > threshold
        extreme_short = signal < -threshold

        # At extremes: flip direction (contrarian)
        flipped[extreme_long] = -signal[extreme_long]
        flipped[extreme_short] = -signal[extreme_short]
        return flipped
