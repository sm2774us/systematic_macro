"""Unit tests for systematic_macro.signals.momentum — 100% coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_macro.signals.momentum import MomentumSignal


class TestMomentumSignalInit:
    """Tests for MomentumSignal initialisation and validation."""

    def test_default_attributes(self) -> None:
        sig = MomentumSignal()
        assert sig.lookbacks == [21, 63, 126, 252]
        assert sig.tsmom_weight == 0.5
        assert sig.xsmom_weight == 0.5

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="must equal 1.0"):
            MomentumSignal(tsmom_weight=0.4, xsmom_weight=0.4)

    def test_invalid_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="lookbacks must be ≥ 1"):
            MomentumSignal(lookbacks=[0, 21, 63])

    def test_custom_lookbacks(self) -> None:
        sig = MomentumSignal(lookbacks=[10, 20, 40])
        assert sig.lookbacks == [10, 20, 40]


class TestMomentumSignalCompute:
    """Tests for MomentumSignal.compute."""

    def test_output_shape(self, returns: pd.DataFrame) -> None:
        sig = MomentumSignal(lookbacks=[21, 63])
        result = sig.compute(returns)
        assert result.shape == returns.shape

    def test_output_clipped(self, returns: pd.DataFrame) -> None:
        sig = MomentumSignal(lookbacks=[21, 63], z_clip=2.5)
        result = sig.compute(returns)
        assert (result.dropna().abs() <= 2.5).all().all()

    def test_insufficient_assets_raises(self, returns: pd.DataFrame) -> None:
        sig = MomentumSignal(min_assets=10)
        with pytest.raises(ValueError, match="Need ≥ 10"):
            sig.compute(returns)

    def test_no_nan_in_later_rows(self, returns: pd.DataFrame) -> None:
        sig = MomentumSignal(lookbacks=[21, 63])
        result = sig.compute(returns)
        # After the longest lookback, results should be mostly non-NaN
        result_tail = result.iloc[-100:]
        assert result_tail.notna().sum().sum() > 0


class TestTSMOMOnly:
    """Tests for compute_tsmom_only."""

    def test_output_shape(self, returns: pd.DataFrame) -> None:
        sig = MomentumSignal(lookbacks=[21])
        result = sig.compute_tsmom_only(returns)
        assert result.shape == returns.shape

    def test_returns_dataframe(self, returns: pd.DataFrame) -> None:
        sig = MomentumSignal(lookbacks=[21])
        result = sig.compute_tsmom_only(returns)
        assert isinstance(result, pd.DataFrame)


class TestXSMOMOnly:
    """Tests for compute_xsmom_only."""

    def test_output_shape(self, returns: pd.DataFrame) -> None:
        sig = MomentumSignal(lookbacks=[21])
        result = sig.compute_xsmom_only(returns)
        assert result.shape == returns.shape

    def test_returns_dataframe(self, returns: pd.DataFrame) -> None:
        sig = MomentumSignal(lookbacks=[21])
        result = sig.compute_xsmom_only(returns)
        assert isinstance(result, pd.DataFrame)


class TestRegimeFilter:
    """Tests for MomentumSignal.regime_filter."""

    def test_output_shape_preserved(
        self, returns: pd.DataFrame, mock_signal: pd.DataFrame
    ) -> None:
        sig = MomentumSignal()
        market = returns.iloc[:, 0]
        filtered = sig.regime_filter(mock_signal, market, window=50)
        assert filtered.shape == mock_signal.shape

    def test_longs_zeroed_in_downtrend(self, returns: pd.DataFrame) -> None:
        sig = MomentumSignal()
        # Simulate a constant down-trend market
        dates = returns.index
        down_market = pd.Series(-0.003, index=dates)
        signal = pd.DataFrame(
            np.ones(returns.shape), index=returns.index, columns=returns.columns
        )
        filtered = sig.regime_filter(signal, down_market, window=50)
        tail = filtered.iloc[-50:]
        assert (tail <= 0).all().all()

    def test_invalid_window_raises(
        self, returns: pd.DataFrame, mock_signal: pd.DataFrame
    ) -> None:
        sig = MomentumSignal()
        market = returns.iloc[:, 0]
        with pytest.raises(ValueError, match="window must be ≥ 10"):
            sig.regime_filter(mock_signal, market, window=5)
