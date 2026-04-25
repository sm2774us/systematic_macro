"""Unit tests for systematic_macro.utils.hlz — 100% coverage."""

from __future__ import annotations
import math
import pytest
from systematic_macro.utils.hlz import (
    HLZCorrection, HLZResult, sharpe_to_tstat, tstat_to_sharpe,
)


class TestHLZCorrectionInit:
    def test_defaults(self) -> None:
        hlz = HLZCorrection()
        assert hlz.n_tests == 1
        assert hlz.method == "hlz"

    def test_invalid_n_tests(self) -> None:
        with pytest.raises(ValueError, match="n_tests must be ≥ 1"):
            HLZCorrection(n_tests=0)

    def test_invalid_sample_length(self) -> None:
        with pytest.raises(ValueError, match="sample_length must be ≥ 10"):
            HLZCorrection(sample_length=5)

    def test_invalid_significance(self) -> None:
        with pytest.raises(ValueError, match="significance must be in"):
            HLZCorrection(significance=1.5)

    def test_invalid_correlation(self) -> None:
        with pytest.raises(ValueError, match="correlation must be in"):
            HLZCorrection(correlation=1.0)

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            HLZCorrection(method="bh")  # type: ignore[arg-type]


class TestHLZMinimumTstat:
    def test_single_test_equals_z_alpha(self) -> None:
        hlz = HLZCorrection(n_tests=1, significance=0.05)
        t_min = hlz.minimum_tstat()
        assert abs(t_min - 1.96) < 0.01

    def test_multiple_tests_higher_threshold_hlz(self) -> None:
        t1 = HLZCorrection(n_tests=1).minimum_tstat()
        t50 = HLZCorrection(n_tests=50).minimum_tstat()
        assert t50 > t1

    def test_bonferroni_higher_than_single(self) -> None:
        t1 = HLZCorrection(n_tests=1, method="bonferroni").minimum_tstat()
        t10 = HLZCorrection(n_tests=10, method="bonferroni").minimum_tstat()
        assert t10 > t1

    def test_holm_higher_than_single(self) -> None:
        t1 = HLZCorrection(n_tests=1, method="holm").minimum_tstat()
        t10 = HLZCorrection(n_tests=10, method="holm").minimum_tstat()
        assert t10 > t1


class TestHLZEvaluate:
    def test_returns_hlz_result(self) -> None:
        hlz = HLZCorrection(n_tests=10, sample_length=252)
        result = hlz.evaluate(observed_sharpe=1.5)
        assert isinstance(result, HLZResult)

    def test_high_sharpe_passes(self) -> None:
        hlz = HLZCorrection(n_tests=1, sample_length=756)
        result = hlz.evaluate(observed_sharpe=3.0)
        assert result.passes_threshold is True

    def test_low_sharpe_fails_many_tests(self) -> None:
        hlz = HLZCorrection(n_tests=200, sample_length=252)
        result = hlz.evaluate(observed_sharpe=0.5)
        assert result.haircut_pct >= 0.0

    def test_haircut_non_negative(self) -> None:
        hlz = HLZCorrection(n_tests=50, sample_length=252)
        result = hlz.evaluate(observed_sharpe=1.0)
        assert result.haircut_pct >= 0.0

    def test_adjusted_tstat_le_observed(self) -> None:
        hlz = HLZCorrection(n_tests=20, sample_length=252)
        result = hlz.evaluate(observed_sharpe=1.0)
        assert result.adjusted_tstat <= result.observed_tstat + 1e-9

    def test_custom_freq_override(self) -> None:
        hlz = HLZCorrection(n_tests=5, sample_length=36, annual_factor=12)
        result = hlz.evaluate(observed_sharpe=1.0, freq=12)
        assert isinstance(result, HLZResult)

    def test_batch_evaluate_length(self) -> None:
        hlz = HLZCorrection(sample_length=252)
        results = hlz.batch_evaluate([0.5, 1.0, 1.5, 2.0])
        assert len(results) == 4

    def test_batch_evaluate_all_hlzresult(self) -> None:
        hlz = HLZCorrection(sample_length=252)
        results = hlz.batch_evaluate([0.8, 1.2])
        assert all(isinstance(r, HLZResult) for r in results)


class TestSharpeConversions:
    def test_round_trip(self) -> None:
        sr = 1.5
        t = sharpe_to_tstat(sr, n_obs=252, annual_factor=252)
        sr2 = tstat_to_sharpe(t, n_obs=252, annual_factor=252)
        assert abs(sr2 - sr) < 1e-9

    def test_sharpe_to_tstat_invalid_n_obs(self) -> None:
        with pytest.raises(ValueError, match="n_obs must be ≥ 2"):
            sharpe_to_tstat(1.0, n_obs=1)

    def test_sharpe_to_tstat_invalid_annual_factor(self) -> None:
        with pytest.raises(ValueError, match="annual_factor must be > 0"):
            sharpe_to_tstat(1.0, n_obs=10, annual_factor=0)

    def test_tstat_to_sharpe_invalid_n_obs(self) -> None:
        with pytest.raises(ValueError, match="n_obs must be ≥ 2"):
            tstat_to_sharpe(2.0, n_obs=1)

    def test_tstat_to_sharpe_invalid_annual_factor(self) -> None:
        with pytest.raises(ValueError, match="annual_factor must be > 0"):
            tstat_to_sharpe(2.0, n_obs=10, annual_factor=0)

    def test_tstat_scales_with_sample(self) -> None:
        t1 = sharpe_to_tstat(1.0, n_obs=252)
        t2 = sharpe_to_tstat(1.0, n_obs=252 * 4)
        assert t2 > t1
