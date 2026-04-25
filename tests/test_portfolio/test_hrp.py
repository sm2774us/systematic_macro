"""Unit tests for systematic_macro.portfolio.hrp — 100% coverage."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from systematic_macro.portfolio.hrp import HRPOptimizer


class TestHRPOptimizerInit:
    def test_defaults(self) -> None:
        hrp = HRPOptimizer()
        assert hrp.cov_method == "ledoit_wolf"
        assert hrp.linkage_method == "ward"
        assert hrp.signal_tilt == 0.3

    def test_invalid_linkage(self) -> None:
        with pytest.raises(ValueError, match="linkage_method must be one of"):
            HRPOptimizer(linkage_method="centroid")

    def test_invalid_signal_tilt(self) -> None:
        with pytest.raises(ValueError, match="signal_tilt must be in"):
            HRPOptimizer(signal_tilt=1.5)

    def test_invalid_max_position(self) -> None:
        with pytest.raises(ValueError, match="max_position must be in"):
            HRPOptimizer(max_position=0.0)


class TestComputeWeights:
    def test_output_is_series(self, returns: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30)
        w = hrp.compute_weights(returns)
        assert isinstance(w, pd.Series)

    def test_weights_sum_to_one(self, returns: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30)
        w = hrp.compute_weights(returns)
        assert abs(float(w.abs().sum()) - 1.0) < 1e-6

    def test_weights_bounded(self, returns: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30, max_position=0.25)
        w = hrp.compute_weights(returns)
        assert (w.abs() <= 0.25 + 1e-9).all()

    def test_all_assets_present(self, returns: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30)
        w = hrp.compute_weights(returns)
        assert set(w.index) == set(returns.columns)

    def test_too_few_assets_raises(self, returns: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30)
        with pytest.raises(ValueError, match="≥ 2"):
            hrp.compute_weights(returns.iloc[:, :1])

    def test_with_signal_tilt(self, returns: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30, signal_tilt=0.3)
        signal = pd.Series(1.0, index=returns.columns)
        w = hrp.compute_weights(returns, signal=signal)
        assert isinstance(w, pd.Series)

    def test_no_signal_tilt_zero(self, returns: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30, signal_tilt=0.0)
        signal = pd.Series(1.0, index=returns.columns)
        w_no_tilt = hrp.compute_weights(returns)
        w_tilt = hrp.compute_weights(returns, signal=signal)
        # With tilt=0, signal has no effect
        pd.testing.assert_series_equal(w_no_tilt, w_tilt)

    def test_different_linkage_methods(self, returns: pd.DataFrame) -> None:
        for method in ["ward", "single", "complete", "average"]:
            hrp = HRPOptimizer(linkage_method=method, min_obs=30)
            w = hrp.compute_weights(returns)
            assert abs(float(w.abs().sum()) - 1.0) < 1e-6


class TestComputeRollingWeights:
    def test_output_shape(self, returns: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30)
        w = hrp.compute_rolling_weights(returns, window=60, step=21)
        assert w.shape == returns.shape

    def test_window_too_small_raises(self, returns: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30)
        with pytest.raises(ValueError, match="min_obs"):
            hrp.compute_rolling_weights(returns, window=10, step=5)

    def test_with_signal_panel(self, returns: pd.DataFrame, mock_signal: pd.DataFrame) -> None:
        hrp = HRPOptimizer(min_obs=30, signal_tilt=0.2)
        w = hrp.compute_rolling_weights(returns, window=60, step=63, signal=mock_signal)
        assert w.shape == returns.shape


class TestCorrToDistance:
    def test_diagonal_is_zero(self, returns: pd.DataFrame) -> None:
        from systematic_macro.utils.covariance import CovarianceEstimator
        corr = CovarianceEstimator(min_obs=30).correlation_matrix(returns)
        dist = HRPOptimizer._corr_to_distance(corr)
        np.testing.assert_allclose(np.diag(dist.values), 0.0, atol=1e-10)

    def test_values_in_range(self, returns: pd.DataFrame) -> None:
        from systematic_macro.utils.covariance import CovarianceEstimator
        corr = CovarianceEstimator(min_obs=30).correlation_matrix(returns)
        dist = HRPOptimizer._corr_to_distance(corr)
        assert (dist.values >= 0).all()
        assert (dist.values <= 1.0 + 1e-9).all()
