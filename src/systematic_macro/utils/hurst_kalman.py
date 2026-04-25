# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Hurst exponent and Kalman filter for signal regime detection and smoothing.

The **Hurst exponent** classifies a time series' memory:
- H ≈ 0.5: random walk (no edge — discard signal).
- H > 0.5: trending / persistent (momentum signal is valid).
- H < 0.5: mean-reverting (carry/contrarian signal is valid).

The **Kalman filter** provides an optimal linear smoother for noisy signals,
adapting its gain in real-time as measurement noise changes — critical for
macro signals where the signal-to-noise ratio shifts across regimes.

Typical usage::

    from systematic_macro.utils.hurst_kalman import HurstEstimator, KalmanFilter1D
    h = HurstEstimator().estimate(price_series)
    smoothed = KalmanFilter1D(process_var=1e-4, obs_var=1e-2).filter(signal_series)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class HurstEstimator:
    """Hurst exponent estimator via Rescaled Range (R/S) analysis.

    Uses the classical R/S method with log-log regression across multiple
    sub-period lengths. Complexity: O(N log N).

    Attributes:
        min_lags: Minimum lag window for R/S calculation.
        max_lags: Maximum lag window (capped at len(series) // 2).
        n_lags: Number of lag points in the log-log regression.
    """

    min_lags: int = 10
    max_lags: int = 500
    n_lags: int = 20

    def __post_init__(self) -> None:
        if self.min_lags < 4:
            raise ValueError(f"min_lags must be ≥ 4, got {self.min_lags}")
        if self.n_lags < 4:
            raise ValueError(f"n_lags must be ≥ 4, got {self.n_lags}")

    def estimate(self, series: pd.Series) -> float:
        """Estimate the Hurst exponent for a univariate series.

        Args:
            series: Price or return time series (NaN-free recommended).

        Returns:
            Hurst exponent H ∈ (0, 1).  Returns 0.5 (random walk) if
            the series is too short or numerically degenerate.

        Raises:
            ValueError: If ``series`` has fewer than ``2 * min_lags`` values.
        """
        clean = series.dropna().to_numpy(dtype=float)
        if len(clean) < 2 * self.min_lags:
            raise ValueError(
                f"Need ≥ {2 * self.min_lags} observations, got {len(clean)}."
            )

        effective_max = min(self.max_lags, len(clean) // 2)
        lags = np.unique(
            np.logspace(
                np.log10(self.min_lags),
                np.log10(effective_max),
                num=self.n_lags,
                dtype=int,
            )
        )

        rs_values: list[float] = []
        for lag in lags:
            rs = self._compute_rs(clean, lag)
            if rs is not None:
                rs_values.append(rs)

        if len(rs_values) < 4:
            logger.warning("Hurst: too few valid R/S points; returning 0.5.")
            return 0.5

        log_lags = np.log(lags[: len(rs_values)])
        log_rs = np.log(np.array(rs_values))
        coeffs = np.polyfit(log_lags, log_rs, deg=1)
        h = float(np.clip(coeffs[0], 0.01, 0.99))
        logger.debug("Hurst H={:.4f} (n={} R/S points)", h, len(rs_values))
        return h

    def estimate_panel(self, prices: pd.DataFrame) -> pd.Series:
        """Estimate Hurst exponent for each column in a price panel.

        Args:
            prices: (T × N) price DataFrame.

        Returns:
            pd.Series of Hurst exponents indexed by ticker.
        """
        results: dict[str, float] = {}
        for col in prices.columns:
            try:
                results[col] = self.estimate(prices[col])
            except ValueError as exc:
                logger.warning("Hurst failed for {}: {}. Using 0.5.", col, exc)
                results[col] = 0.5
        return pd.Series(results, name="hurst")

    def classify(self, h: float) -> str:
        """Classify a Hurst value into a regime label.

        Args:
            h: Hurst exponent ∈ (0, 1).

        Returns:
            ``"trending"`` (H > 0.55), ``"mean_reverting"`` (H < 0.45),
            or ``"random_walk"`` (0.45 ≤ H ≤ 0.55).
        """
        if h > 0.55:
            return "trending"
        if h < 0.45:
            return "mean_reverting"
        return "random_walk"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rs(series: np.ndarray, lag: int) -> float | None:
        """Compute mean R/S statistic for a given lag.

        Splits series into non-overlapping sub-periods of length ``lag``,
        computes R/S for each, and returns the mean.

        Args:
            series: 1-D float array.
            lag: Sub-period length.

        Returns:
            Mean R/S value or ``None`` if degenerate.
        """
        n_chunks = len(series) // lag
        if n_chunks < 1:
            return None

        rs_list: list[float] = []
        for i in range(n_chunks):
            chunk = series[i * lag : (i + 1) * lag]
            mean = np.mean(chunk)
            deviations = np.cumsum(chunk - mean)
            r = float(np.max(deviations) - np.min(deviations))
            s = float(np.std(chunk, ddof=1))
            if s > 0:
                rs_list.append(r / s)

        return float(np.mean(rs_list)) if rs_list else None


@dataclass
class KalmanFilter1D:
    """Scalar Kalman filter for adaptive signal smoothing.

    Implements the standard 1-D linear Kalman filter with constant
    process and observation noise covariances. The filter adapts its
    gain automatically: when obs noise is high relative to process noise
    it trusts prior state; when low it trusts the measurement.

    Attributes:
        process_var: State transition noise variance (Q).
            Larger = faster adaptation.
        obs_var: Observation noise variance (R).
            Larger = more smoothing.
        init_state: Initial state estimate (default 0.0).
        init_cov: Initial state covariance (default 1.0).
    """

    process_var: float = 1e-4
    obs_var: float = 1e-2
    init_state: float = 0.0
    init_cov: float = 1.0

    def __post_init__(self) -> None:
        if self.process_var <= 0:
            raise ValueError(f"process_var must be > 0, got {self.process_var}")
        if self.obs_var <= 0:
            raise ValueError(f"obs_var must be > 0, got {self.obs_var}")

    def filter(self, signal: pd.Series) -> pd.Series:
        """Apply Kalman filter to a univariate signal series.

        Args:
            signal: Noisy signal observations (NaN are forward-propagated
                using the prior estimate, not updated).

        Returns:
            pd.Series of smoothed state estimates, same index as ``signal``.
        """
        x = self.init_state
        p = self.init_cov
        smoothed = np.empty(len(signal), dtype=float)

        for i, obs in enumerate(signal.to_numpy(dtype=float)):
            # Predict
            x_pred = x
            p_pred = p + self.process_var

            if np.isnan(obs):
                # No update step — propagate prediction
                smoothed[i] = x_pred
                x, p = x_pred, p_pred
            else:
                # Kalman gain
                k = p_pred / (p_pred + self.obs_var)
                # Update
                x = x_pred + k * (obs - x_pred)
                p = (1.0 - k) * p_pred
                smoothed[i] = x

        return pd.Series(smoothed, index=signal.index, name=signal.name)

    def filter_panel(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Apply Kalman filter column-wise to a signal panel.

        Args:
            signals: (T × N) signal DataFrame.

        Returns:
            (T × N) smoothed signal DataFrame.
        """
        return pd.DataFrame(
            {col: self.filter(signals[col]) for col in signals.columns},
            index=signals.index,
        )

    def adaptive_filter(
        self,
        signal: pd.Series,
        window: int = 63,
    ) -> pd.Series:
        """Adaptive Kalman filter: re-estimates obs_var from rolling window.

        Uses a rolling window to estimate the current observation noise and
        updates the Kalman gain accordingly. More responsive in low-noise
        regimes, more stable in high-noise regimes.

        Args:
            signal: Input signal.
            window: Rolling window for obs variance estimation.

        Returns:
            Adaptively smoothed signal.

        Raises:
            ValueError: If ``window`` < 5.
        """
        if window < 5:
            raise ValueError(f"window must be ≥ 5, got {window}")

        rolling_var = signal.rolling(window, min_periods=5).var().fillna(self.obs_var)
        x = self.init_state
        p = self.init_cov
        smoothed = np.empty(len(signal), dtype=float)

        obs_array = signal.to_numpy(dtype=float)
        var_array = rolling_var.to_numpy(dtype=float)

        for i in range(len(obs_array)):
            x_pred = x
            p_pred = p + self.process_var
            obs = obs_array[i]
            r_local = float(var_array[i]) if not np.isnan(var_array[i]) else self.obs_var

            if np.isnan(obs):
                smoothed[i] = x_pred
                x, p = x_pred, p_pred
            else:
                k = p_pred / (p_pred + max(r_local, 1e-12))
                x = x_pred + k * (obs - x_pred)
                p = (1.0 - k) * p_pred
                smoothed[i] = x

        return pd.Series(smoothed, index=signal.index, name=signal.name)
