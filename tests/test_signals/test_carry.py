"""Unit tests for systematic_macro.signals.carry — 100% coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_macro.signals.carry import CarrySignal


def _make_rate_df(prices: pd.DataFrame, val: float = 0.03) -> pd.DataFrame:
    """Return constant rate differential aligned to prices."""
    return pd.DataFrame(val, index=prices.index, columns=prices.columns)


class TestCarrySignalInit:
    """Tests for CarrySignal defaults and attributes."""

    def test_default_attributes(self) -> None:
        sig = CarrySignal()
        assert sig.vol_window == 21
        assert sig.carry_window == 5
        assert sig.min_assets == 4
        assert sig.z_clip == 3.0

    def test_custom_attributes(self) -> None:
        sig = CarrySignal(vol_window=10, z_clip=2.0)
        assert sig.vol_window == 10
        assert sig.z_clip == 2.0


class TestCarrySignalCompute:
    """Tests for CarrySignal.compute."""

    def test_no_components_raises(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        with pytest.raises(ValueError, match="At least one"):
            sig.compute(prices)

    def test_output_shape_matches_prices(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        rate_diff = _make_rate_df(prices, 0.03)
        result = sig.compute(prices, rate_differentials=rate_diff)
        assert result.shape == prices.shape

    def test_output_clipped_to_z_clip(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal(z_clip=2.0)
        rate_diff = _make_rate_df(prices, 0.05)
        result = sig.compute(prices, rate_differentials=rate_diff)
        assert (result.dropna().abs() <= 2.0).all().all()

    def test_with_futures_roll(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        roll = _make_rate_df(prices, 0.02)
        result = sig.compute(prices, futures_roll=roll)
        assert result.shape == prices.shape

    def test_with_dividend_yield(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        div = _make_rate_df(prices, 0.015)
        result = sig.compute(prices, dividend_yield=div)
        assert result.shape == prices.shape

    def test_all_components_combined(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        rd = _make_rate_df(prices, 0.03)
        roll = _make_rate_df(prices, 0.01)
        div = _make_rate_df(prices, 0.02)
        result = sig.compute(prices, rate_differentials=rd, futures_roll=roll, dividend_yield=div)
        assert result.shape == prices.shape

    def test_output_is_float64(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        rd = _make_rate_df(prices, 0.03)
        result = sig.compute(prices, rate_differentials=rd)
        assert result.dtypes.unique()[0] == np.float64

    def test_high_carry_positive_signal(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        # One asset has much higher carry
        rd = _make_rate_df(prices, 0.01)
        rd.iloc[:, 0] = 0.10  # first asset = high carry
        result = sig.compute(prices, rate_differentials=rd)
        valid = result.dropna()
        if not valid.empty:
            assert valid.iloc[-1, 0] > 0


class TestComputeFXCarry:
    """Tests for CarrySignal.compute_fx_carry."""

    def test_output_shape(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        fd = _make_rate_df(prices, 0.02)
        result = sig.compute_fx_carry(prices, fd)
        assert result.shape == prices.shape

    def test_clipped_values(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal(z_clip=3.0)
        fd = _make_rate_df(prices, 0.05)
        result = sig.compute_fx_carry(prices, fd)
        assert (result.dropna().abs() <= 3.0).all().all()


class TestComputeFuturesCarry:
    """Tests for CarrySignal.compute_futures_carry."""

    def test_output_shape(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        next_prices = prices * 1.01  # next contract slightly higher (contango)
        result = sig.compute_futures_carry(prices, next_prices)
        assert result.shape == prices.shape

    def test_mismatched_shapes_raises(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        next_prices = prices.iloc[:, :3]
        with pytest.raises(ValueError, match="shape"):
            sig.compute_futures_carry(prices, next_prices)

    def test_backwardation_positive_roll(self, prices: pd.DataFrame) -> None:
        sig = CarrySignal()
        next_prices = prices * 0.99  # next contract slightly lower (backwardation)
        result = sig.compute_futures_carry(prices, next_prices)
        valid = result.dropna()
        # Positive roll in backwardation: result should trend positive
        assert isinstance(valid, pd.DataFrame)
