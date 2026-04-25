"""Unit tests for systematic_macro.backtest.engine — 100% coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_macro.backtest.engine import (
    BacktestResult,
    FoldResult,
    WalkForwardEngine,
)


class TestWalkForwardEngineInit:
    """Tests for WalkForwardEngine initialisation and validation."""

    def test_defaults(self) -> None:
        eng = WalkForwardEngine()
        assert eng.is_years == 5
        assert eng.oos_years == 1
        assert eng.step_months == 6

    def test_invalid_is_years_raises(self) -> None:
        with pytest.raises(ValueError, match="is_years must be ≥ 1"):
            WalkForwardEngine(is_years=0)

    def test_invalid_oos_years_raises(self) -> None:
        with pytest.raises(ValueError, match="oos_years must be ≥ 1"):
            WalkForwardEngine(oos_years=0)

    def test_invalid_step_months_raises(self) -> None:
        with pytest.raises(ValueError, match="step_months must be ≥ 1"):
            WalkForwardEngine(step_months=0)

    def test_invalid_tc_bps_raises(self) -> None:
        with pytest.raises(ValueError, match="tc_bps must be ≥ 0"):
            WalkForwardEngine(tc_bps=-1.0)

    def test_bonferroni_adjustment(self) -> None:
        eng = WalkForwardEngine(n_tests=20, sharpe_threshold=0.5)
        # Threshold should have been increased
        assert eng.sharpe_threshold > 0.5

    def test_no_bonferroni_for_single_test(self) -> None:
        eng = WalkForwardEngine(n_tests=1, sharpe_threshold=0.5)
        assert eng.sharpe_threshold == pytest.approx(0.5, abs=1e-9)


class TestGenerateFolds:
    """Tests for WalkForwardEngine._generate_folds."""

    def test_sufficient_data_generates_folds(self, returns: pd.DataFrame) -> None:
        eng = WalkForwardEngine(is_years=3, oos_years=1, step_months=6)
        folds = eng._generate_folds(returns.index)
        assert len(folds) > 0

    def test_insufficient_data_returns_empty(self, short_returns: pd.DataFrame) -> None:
        eng = WalkForwardEngine(is_years=5, oos_years=2, step_months=6)
        folds = eng._generate_folds(short_returns.index)
        assert folds == []

    def test_fold_dates_are_ordered(self, returns: pd.DataFrame) -> None:
        eng = WalkForwardEngine(is_years=2, oos_years=1, step_months=6)
        folds = eng._generate_folds(returns.index)
        for is_start, is_end, oos_start, oos_end in folds:
            assert is_start <= is_end
            assert is_end <= oos_start
            assert oos_start <= oos_end


@pytest.fixture(scope="module")
def fast_engine() -> WalkForwardEngine:
    """Fast engine with small IS/OOS for unit tests."""
    return WalkForwardEngine(
        is_years=1, oos_years=1, step_months=6,
        icir_threshold=0.05, sharpe_threshold=0.05,
        tc_bps=1.0, n_tests=1,
    )


@pytest.fixture(scope="module")
def fast_result(
    fast_engine: WalkForwardEngine,
    mock_signal: pd.DataFrame,
    returns: pd.DataFrame,
) -> BacktestResult:
    """Pre-computed BacktestResult for fast_engine."""
    return fast_engine.run(mock_signal, returns)


class TestWalkForwardRun:
    """Tests for WalkForwardEngine.run."""

    def test_run_returns_backtest_result(self, fast_result: BacktestResult) -> None:
        assert isinstance(fast_result, BacktestResult)

    def test_result_has_folds(self, fast_result: BacktestResult) -> None:
        assert len(fast_result.folds) > 0

    def test_oos_returns_is_series(self, fast_result: BacktestResult) -> None:
        assert isinstance(fast_result.oos_returns, pd.Series)

    def test_mismatched_columns_raises(
        self, fast_engine: WalkForwardEngine,
        mock_signal: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> None:
        bad_signal = mock_signal.copy()
        bad_signal.columns = [f"Z{i}" for i in range(bad_signal.shape[1])]
        with pytest.raises(ValueError, match="identical columns"):
            fast_engine.run(bad_signal, returns)

    def test_insufficient_history_raises(
        self, fast_engine: WalkForwardEngine,
        mock_signal: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> None:
        eng = WalkForwardEngine(is_years=10, oos_years=5, step_months=6)
        with pytest.raises(ValueError, match="Insufficient history"):
            eng.run(mock_signal.iloc[:80], returns.iloc[:80])

    def test_with_portfolio_weights(
        self, fast_engine: WalkForwardEngine,
        mock_signal: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> None:
        weights = mock_signal.div(
            mock_signal.abs().sum(axis=1), axis=0
        ).fillna(0)
        result = fast_engine.run(mock_signal, returns, portfolio_weights=weights)
        assert isinstance(result, BacktestResult)


class TestBacktestResultSummary:
    """Tests for BacktestResult.summary."""

    def test_summary_keys(self, fast_result: BacktestResult) -> None:
        summary = fast_result.summary()
        expected_keys = {
            "n_folds", "n_passed", "pass_rate",
            "oos_sharpe", "oos_mdd", "oos_icir",
            "is_icir_mean", "is_icir_std",
        }
        assert expected_keys.issubset(summary.keys())

    def test_pass_rate_in_range(self, fast_result: BacktestResult) -> None:
        assert 0.0 <= fast_result.summary()["pass_rate"] <= 1.0


class TestPassesPortfolioGate:
    """Tests for BacktestResult.passes_portfolio_gate."""

    def test_gate_returns_bool(self, fast_result: BacktestResult) -> None:
        rng = np.random.default_rng(99)
        portfolio = pd.Series(
            rng.normal(0.001, 0.01, len(fast_result.oos_returns)),
            index=fast_result.oos_returns.index,
        )
        gate = fast_result.passes_portfolio_gate(portfolio)
        assert isinstance(gate, bool)

    def test_identical_signal_fails_both_gates(
        self, fast_result: BacktestResult,
    ) -> None:
        identical = fast_result.oos_returns.copy()
        gate = fast_result.passes_portfolio_gate(
            identical,
            delta_sharpe_threshold=0.05,
            correlation_threshold=0.6,
        )
        assert not gate


class TestMonitorLive:
    """Tests for WalkForwardEngine.monitor_live."""

    def test_output_columns(
        self, fast_engine: WalkForwardEngine,
        mock_signal: pd.DataFrame, returns: pd.DataFrame,
    ) -> None:
        monitor = fast_engine.monitor_live(mock_signal, returns, is_ic_baseline=0.3)
        assert "rolling_ic" in monitor.columns
        assert "ic_ratio" in monitor.columns
        assert "flagged" in monitor.columns

    def test_zero_baseline_raises(
        self, fast_engine: WalkForwardEngine,
        mock_signal: pd.DataFrame, returns: pd.DataFrame,
    ) -> None:
        with pytest.raises(ValueError, match="is_ic_baseline must be non-zero"):
            fast_engine.monitor_live(mock_signal, returns, is_ic_baseline=0.0)

    def test_flagged_is_boolean(
        self, fast_engine: WalkForwardEngine,
        mock_signal: pd.DataFrame, returns: pd.DataFrame,
    ) -> None:
        monitor = fast_engine.monitor_live(mock_signal, returns, is_ic_baseline=0.5)
        assert monitor["flagged"].dtype == bool


class TestFoldResult:
    """Tests for FoldResult named tuple."""

    def test_construction(self) -> None:
        fr = FoldResult(
            fold_id=0,
            is_start=pd.Timestamp("2018-01-01"),
            is_end=pd.Timestamp("2022-12-31"),
            oos_start=pd.Timestamp("2023-01-01"),
            oos_end=pd.Timestamp("2023-12-31"),
            is_icir=0.6,
            oos_icir=0.55,
            oos_sharpe=0.7,
            oos_mdd=0.08,
            passed_gate=True,
        )
        assert fr.passed_gate is True
        assert fr.fold_id == 0
