"""Unit tests for systematic_macro.utils.hurst_kalman — 100% coverage."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from systematic_macro.utils.hurst_kalman import HurstEstimator, KalmanFilter1D


class TestHurstEstimatorInit:
    def test_defaults(self) -> None:
        h = HurstEstimator()
        assert h.min_lags == 10 and h.n_lags == 20

    def test_invalid_min_lags(self) -> None:
        with pytest.raises(ValueError, match="min_lags must be ≥ 4"):
            HurstEstimator(min_lags=3)

    def test_invalid_n_lags(self) -> None:
        with pytest.raises(ValueError, match="n_lags must be ≥ 4"):
            HurstEstimator(n_lags=3)


class TestHurstEstimate:
    def test_trending_series_high_hurst(self) -> None:
        rng = np.random.default_rng(0)
        # Fractional Brownian motion approximation with strong trend
        prices = pd.Series(np.cumsum(np.abs(rng.standard_normal(500))) + 100)
        h = HurstEstimator(min_lags=5, n_lags=10).estimate(prices)
        assert 0.0 < h < 1.0

    def test_mean_reverting_low_hurst(self) -> None:
        rng = np.random.default_rng(1)
        # AR(1) with negative autocorrelation → mean reverting
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = -0.5 * x[i - 1] + rng.standard_normal()
        h = HurstEstimator(min_lags=5, n_lags=10).estimate(pd.Series(x))
        assert h < 0.5

    def test_insufficient_obs_raises(self) -> None:
        with pytest.raises(ValueError, match="≥"):
            HurstEstimator(min_lags=10).estimate(pd.Series([1.0, 2.0]))

    def test_output_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        series = pd.Series(rng.standard_normal(300))
        h = HurstEstimator(min_lags=5, n_lags=8).estimate(series)
        assert 0.0 < h < 1.0

    def test_returns_float(self) -> None:
        series = pd.Series(np.cumsum(np.random.default_rng(7).standard_normal(200)))
        h = HurstEstimator(min_lags=5, n_lags=8).estimate(series)
        assert isinstance(h, float)


class TestHurstEstimatePanel:
    def test_panel_returns_series(self, prices: pd.DataFrame) -> None:
        est = HurstEstimator(min_lags=5, n_lags=8)
        result = est.estimate_panel(prices)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(prices.columns)

    def test_panel_all_in_unit_interval(self, prices: pd.DataFrame) -> None:
        est = HurstEstimator(min_lags=5, n_lags=8)
        result = est.estimate_panel(prices)
        assert ((result > 0) & (result < 1)).all()


class TestHurstClassify:
    def test_trending(self) -> None:
        assert HurstEstimator().classify(0.7) == "trending"

    def test_mean_reverting(self) -> None:
        assert HurstEstimator().classify(0.3) == "mean_reverting"

    def test_random_walk(self) -> None:
        assert HurstEstimator().classify(0.5) == "random_walk"

    def test_boundary_trending(self) -> None:
        assert HurstEstimator().classify(0.56) == "trending"

    def test_boundary_reverting(self) -> None:
        assert HurstEstimator().classify(0.44) == "mean_reverting"


class TestKalmanFilter1DInit:
    def test_defaults(self) -> None:
        kf = KalmanFilter1D()
        assert kf.process_var == 1e-4
        assert kf.obs_var == 1e-2

    def test_invalid_process_var(self) -> None:
        with pytest.raises(ValueError, match="process_var must be > 0"):
            KalmanFilter1D(process_var=0.0)

    def test_invalid_obs_var(self) -> None:
        with pytest.raises(ValueError, match="obs_var must be > 0"):
            KalmanFilter1D(obs_var=-1.0)


class TestKalmanFilter:
    def test_output_length(self) -> None:
        kf = KalmanFilter1D()
        sig = pd.Series(np.random.default_rng(0).standard_normal(100))
        result = kf.filter(sig)
        assert len(result) == 100

    def test_output_is_series(self) -> None:
        kf = KalmanFilter1D()
        sig = pd.Series(np.ones(50))
        result = kf.filter(sig)
        assert isinstance(result, pd.Series)

    def test_smooth_constant_signal(self) -> None:
        kf = KalmanFilter1D(process_var=1e-6, obs_var=1e-4)
        sig = pd.Series(np.ones(200))
        result = kf.filter(sig)
        assert abs(float(result.iloc[-1]) - 1.0) < 0.05

    def test_handles_nans(self) -> None:
        kf = KalmanFilter1D()
        sig = pd.Series([1.0, np.nan, 1.0, np.nan, 1.0])
        result = kf.filter(sig)
        assert not result.isna().any()

    def test_preserves_index(self) -> None:
        kf = KalmanFilter1D()
        dates = pd.bdate_range("2020-01-01", periods=50)
        sig = pd.Series(np.ones(50), index=dates)
        result = kf.filter(sig)
        assert result.index.equals(dates)

    def test_panel_filter_shape(self, mock_signal: pd.DataFrame) -> None:
        kf = KalmanFilter1D()
        result = kf.filter_panel(mock_signal)
        assert result.shape == mock_signal.shape

    def test_adaptive_filter_output_shape(self) -> None:
        kf = KalmanFilter1D()
        sig = pd.Series(np.random.default_rng(3).standard_normal(200))
        result = kf.adaptive_filter(sig, window=20)
        assert len(result) == 200

    def test_adaptive_filter_invalid_window(self) -> None:
        kf = KalmanFilter1D()
        sig = pd.Series(np.ones(50))
        with pytest.raises(ValueError, match="window must be ≥ 5"):
            kf.adaptive_filter(sig, window=4)
