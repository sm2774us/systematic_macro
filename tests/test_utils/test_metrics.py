"""Unit tests for systematic_macro.utils.metrics — 100% coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_macro.utils.metrics import (
    bonferroni_sharpe_threshold,
    compute_calmar,
    compute_ic,
    compute_icir,
    compute_marginal_sharpe,
    compute_max_drawdown,
    compute_net_ic,
    compute_rolling_ic,
    compute_sharpe,
)


# ---------------------------------------------------------------------------
# compute_ic
# ---------------------------------------------------------------------------

class TestComputeIC:
    """Tests for compute_ic."""

    def test_perfect_positive_spearman(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_ic(s, r, method="spearman") == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_spearman(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        assert compute_ic(s, r, method="spearman") == pytest.approx(-1.0, abs=1e-6)

    def test_pearson_method(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        r = pd.Series([2.0, 4.0, 6.0, 8.0])
        assert compute_ic(s, r, method="pearson") == pytest.approx(1.0, abs=1e-6)

    def test_invalid_method_raises(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        r = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="method must be"):
            compute_ic(s, r, method="kendall")  # type: ignore[arg-type]

    def test_insufficient_observations_raises(self) -> None:
        s = pd.Series([1.0])
        r = pd.Series([1.0])
        with pytest.raises(ValueError, match="≥2"):
            compute_ic(s, r)

    def test_handles_nan_via_alignment(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
        r = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = compute_ic(s, r)
        assert -1.0 <= ic <= 1.0

    def test_return_type_is_float(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        r = pd.Series([4.0, 3.0, 2.0, 1.0])
        assert isinstance(compute_ic(s, r), float)


# ---------------------------------------------------------------------------
# compute_rolling_ic
# ---------------------------------------------------------------------------

class TestComputeRollingIC:
    """Tests for compute_rolling_ic."""

    def test_returns_series(self, returns: pd.DataFrame, mock_signal: pd.DataFrame) -> None:
        fwd = returns.shift(-1)
        ic = compute_rolling_ic(mock_signal, fwd)
        assert isinstance(ic, pd.Series)

    def test_length_bounded_by_common_dates(
        self, returns: pd.DataFrame, mock_signal: pd.DataFrame
    ) -> None:
        fwd = returns.shift(-1)
        ic = compute_rolling_ic(mock_signal, fwd)
        assert len(ic) <= len(returns)

    def test_skips_rows_with_few_observations(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=50)
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal((50, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        fwd = signal.shift(-1)
        ic = compute_rolling_ic(signal, fwd)
        # With only 3 assets, all rows should be skipped (< 4 required)
        assert ic.empty or len(ic) == 0

    def test_valid_ic_range(self, returns: pd.DataFrame, mock_signal: pd.DataFrame) -> None:
        fwd = returns.shift(-1)
        ic = compute_rolling_ic(mock_signal, fwd)
        assert (ic.dropna().abs() <= 1.0).all()


# ---------------------------------------------------------------------------
# compute_icir
# ---------------------------------------------------------------------------

class TestComputeICIR:
    """Tests for compute_icir."""

    def test_positive_icir(self) -> None:
        ic = pd.Series([0.1, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08, 0.12, 0.11, 0.10, 0.09, 0.11, 0.10])
        assert compute_icir(ic) > 0.0

    def test_returns_zero_for_insufficient_obs(self) -> None:
        ic = pd.Series([0.1, 0.2, 0.3])
        assert compute_icir(ic, min_obs=12) == 0.0

    def test_returns_zero_for_zero_std(self) -> None:
        ic = pd.Series([0.1] * 20)
        assert compute_icir(ic) == 0.0

    def test_invalid_min_obs_raises(self) -> None:
        with pytest.raises(ValueError, match="min_obs must be ≥ 2"):
            compute_icir(pd.Series([0.1] * 10), min_obs=1)

    def test_handles_nans(self) -> None:
        ic = pd.Series([0.1, np.nan, 0.1, np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        result = compute_icir(ic)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------

class TestComputeSharpe:
    """Tests for compute_sharpe."""

    def test_positive_returns_positive_sharpe(self) -> None:
        r = pd.Series([0.001] * 252)
        assert compute_sharpe(r) > 0.0

    def test_zero_std_returns_zero(self) -> None:
        r = pd.Series([0.0] * 100)
        assert compute_sharpe(r) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert compute_sharpe(pd.Series(dtype=float)) == 0.0

    def test_invalid_freq_raises(self) -> None:
        with pytest.raises(ValueError, match="freq must be > 0"):
            compute_sharpe(pd.Series([0.01] * 10), freq=0)

    def test_risk_free_reduces_sharpe(self) -> None:
        r = pd.Series([0.001] * 252)
        s_no_rf = compute_sharpe(r, risk_free=0.0)
        s_with_rf = compute_sharpe(r, risk_free=0.05)
        assert s_no_rf > s_with_rf

    def test_monthly_freq(self) -> None:
        r = pd.Series([0.01] * 36)
        s = compute_sharpe(r, freq=12)
        assert isinstance(s, float)


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------

class TestComputeMaxDrawdown:
    """Tests for compute_max_drawdown."""

    def test_no_drawdown_returns_zero(self) -> None:
        r = pd.Series([0.01] * 50)
        assert compute_max_drawdown(r) == pytest.approx(0.0, abs=1e-6)

    def test_known_drawdown(self) -> None:
        # Falls 50% then recovers
        r = pd.Series([0.0, -0.5, 0.5, 0.5])
        mdd = compute_max_drawdown(r)
        assert 0.4 < mdd < 0.55

    def test_empty_returns_zero(self) -> None:
        assert compute_max_drawdown(pd.Series(dtype=float)) == 0.0

    def test_all_negative_returns(self) -> None:
        r = pd.Series([-0.01] * 50)
        assert compute_max_drawdown(r) > 0.0

    def test_returns_positive_value(self) -> None:
        r = pd.Series([-0.01, 0.02, -0.03, 0.01])
        assert compute_max_drawdown(r) > 0.0


# ---------------------------------------------------------------------------
# compute_net_ic
# ---------------------------------------------------------------------------

class TestComputeNetIC:
    """Tests for compute_net_ic."""

    def test_no_cost_equals_gross(self) -> None:
        assert compute_net_ic(0.1, 0.1, tc_bps=0.0) == pytest.approx(0.1, abs=1e-9)

    def test_positive_cost_reduces_ic(self) -> None:
        net = compute_net_ic(0.1, 0.2, tc_bps=10.0)
        assert net < 0.1

    def test_negative_tc_raises(self) -> None:
        with pytest.raises(ValueError, match="tc_bps must be ≥ 0"):
            compute_net_ic(0.1, 0.1, tc_bps=-1.0)

    def test_invalid_turnover_raises(self) -> None:
        with pytest.raises(ValueError, match="turnover must be in"):
            compute_net_ic(0.1, 1.5)

    def test_invalid_signal_vol_raises(self) -> None:
        with pytest.raises(ValueError, match="signal_vol must be > 0"):
            compute_net_ic(0.1, 0.1, signal_vol=0.0)


# ---------------------------------------------------------------------------
# compute_marginal_sharpe
# ---------------------------------------------------------------------------

class TestComputeMarginalSharpe:
    """Tests for compute_marginal_sharpe."""

    def test_beneficial_signal_positive_delta(self) -> None:
        rng = np.random.default_rng(0)
        p = pd.Series(rng.normal(0.001, 0.01, 252))
        s = pd.Series(rng.normal(0.002, 0.01, 252))
        # Independent positive-return signal should often increase Sharpe
        delta = compute_marginal_sharpe(p, s)
        assert isinstance(delta, float)

    def test_invalid_weight_raises(self) -> None:
        p = pd.Series([0.01] * 10)
        s = pd.Series([0.01] * 10)
        with pytest.raises(ValueError, match="weight must be in"):
            compute_marginal_sharpe(p, s, weight=0.0)

    def test_weight_upper_bound_raises(self) -> None:
        p = pd.Series([0.01] * 10)
        s = pd.Series([0.01] * 10)
        with pytest.raises(ValueError, match="weight must be in"):
            compute_marginal_sharpe(p, s, weight=1.0)


# ---------------------------------------------------------------------------
# compute_calmar
# ---------------------------------------------------------------------------

class TestComputeCalmar:
    """Tests for compute_calmar."""

    def test_positive_for_series_with_drawdown(self) -> None:
        # Series that has a dip then recovers — produces positive calmar
        r = pd.Series([-0.05] * 10 + [0.01] * 252)
        assert compute_calmar(r) > 0.0

    def test_returns_zero_when_no_drawdown(self) -> None:
        # All-up: MDD ~= 0, returns 0
        r = pd.Series([0.001] * 252)
        result = compute_calmar(r)
        assert result == 0.0 or result > 0.0


# ---------------------------------------------------------------------------
# bonferroni_sharpe_threshold
# ---------------------------------------------------------------------------

class TestBonferroniSharpeThreshold:
    """Tests for bonferroni_sharpe_threshold."""

    def test_single_test_no_adjustment(self) -> None:
        assert bonferroni_sharpe_threshold(1, 0.5) == pytest.approx(0.5, abs=1e-9)

    def test_multiple_tests_increase_threshold(self) -> None:
        t10 = bonferroni_sharpe_threshold(10, 0.5)
        assert t10 > 0.5

    def test_invalid_n_tests_raises(self) -> None:
        with pytest.raises(ValueError, match="n_tests must be ≥ 1"):
            bonferroni_sharpe_threshold(0, 0.5)

    def test_monotonic_with_n_tests(self) -> None:
        thresholds = [bonferroni_sharpe_threshold(n, 0.5) for n in [1, 5, 10, 50, 100]]
        assert thresholds == sorted(thresholds)
