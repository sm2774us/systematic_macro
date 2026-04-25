"""Unit tests for systematic_macro.utils.covariance — 100% coverage."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from systematic_macro.utils.covariance import CovarianceEstimator


class TestCovarianceEstimatorInit:
    def test_defaults(self) -> None:
        est = CovarianceEstimator()
        assert est.method == "ledoit_wolf"
        assert est.annualise is True

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            CovarianceEstimator(method="shrunk")  # type: ignore[arg-type]

    def test_invalid_annual_factor(self) -> None:
        with pytest.raises(ValueError, match="annual_factor must be > 0"):
            CovarianceEstimator(annual_factor=0)


class TestFit:
    def test_output_shape(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(method="ledoit_wolf", min_obs=30)
        cov = est.fit(returns)
        n = returns.shape[1]
        assert cov.shape == (n, n)

    def test_symmetric(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(method="ledoit_wolf", min_obs=30)
        cov = est.fit(returns)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)

    def test_positive_diagonal(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=30)
        cov = est.fit(returns)
        assert (np.diag(cov.values) > 0).all()

    def test_oas_method(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(method="oas", min_obs=30)
        cov = est.fit(returns)
        assert cov.shape == (returns.shape[1], returns.shape[1])

    def test_sample_method(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(method="sample", min_obs=30, annualise=False)
        cov = est.fit(returns)
        assert cov.shape == (returns.shape[1], returns.shape[1])

    def test_insufficient_obs_raises(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=5000)
        with pytest.raises(ValueError, match="≥"):
            est.fit(returns)

    def test_annualise_scales_by_factor(self, returns: pd.DataFrame) -> None:
        ann = CovarianceEstimator(annualise=True, annual_factor=252, min_obs=30)
        raw = CovarianceEstimator(annualise=False, min_obs=30)
        cov_ann = ann.fit(returns)
        cov_raw = raw.fit(returns)
        ratio = cov_ann.values / cov_raw.values
        np.testing.assert_allclose(ratio, np.full_like(ratio, 252.0), rtol=0.01)


class TestPrecisionMatrix:
    def test_shape(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=30)
        prec = est.precision_matrix(returns)
        n = returns.shape[1]
        assert prec.shape == (n, n)

    def test_invertible(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=30)
        prec = est.precision_matrix(returns)
        assert np.isfinite(prec.values).all()


class TestShrinkageIntensity:
    def test_in_range(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=30)
        alpha = est.shrinkage_intensity(returns)
        assert 0.0 <= alpha <= 1.0

    def test_sample_returns_zero(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(method="sample", min_obs=30)
        assert est.shrinkage_intensity(returns) == 0.0

    def test_insufficient_obs_raises(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=50000)
        with pytest.raises(ValueError, match="≥"):
            est.shrinkage_intensity(returns)


class TestCorrelationMatrix:
    def test_diagonal_ones(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=30)
        corr = est.correlation_matrix(returns)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-6)

    def test_values_in_range(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=30)
        corr = est.correlation_matrix(returns)
        assert (corr.values >= -1.0 - 1e-9).all()
        assert (corr.values <= 1.0 + 1e-9).all()


class TestEffectiveRank:
    def test_positive(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=30)
        rank = est.effective_rank(returns)
        assert rank >= 1.0

    def test_bounded_by_n(self, returns: pd.DataFrame) -> None:
        est = CovarianceEstimator(min_obs=30)
        rank = est.effective_rank(returns)
        assert rank <= returns.shape[1] + 1e-6
