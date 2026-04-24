"""Unit tests for systematic_macro.signals.flow — 100% coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_macro.signals.flow import FlowSignal


def _make_volume(prices: pd.DataFrame, scale: float = 1e6) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        rng.uniform(0.5, 1.5, prices.shape) * scale,
        index=prices.index,
        columns=prices.columns,
    )


class TestFlowSignalInit:
    """Tests for FlowSignal initialisation."""

    def test_defaults(self) -> None:
        sig = FlowSignal()
        assert sig.cot_window == 52
        assert sig.flow_window == 21
        assert sig.contrarian is True
        assert sig.contrarian_threshold == 1.5

    def test_invalid_cot_window_raises(self) -> None:
        with pytest.raises(ValueError, match="cot_window must be ≥ 4"):
            FlowSignal(cot_window=3)

    def test_invalid_flow_window_raises(self) -> None:
        with pytest.raises(ValueError, match="flow_window must be ≥ 5"):
            FlowSignal(flow_window=4)


class TestFlowSignalCompute:
    """Tests for FlowSignal.compute."""

    def test_output_shape_with_volume(self, prices: pd.DataFrame) -> None:
        sig = FlowSignal()
        vol = _make_volume(prices)
        result = sig.compute(prices, volume=vol)
        assert result.shape == prices.shape

    def test_output_shape_no_volume(self, prices: pd.DataFrame) -> None:
        sig = FlowSignal()
        result = sig.compute(prices)
        assert result.shape == prices.shape

    def test_output_clipped(self, prices: pd.DataFrame) -> None:
        sig = FlowSignal(z_clip=2.0)
        result = sig.compute(prices)
        assert (result.dropna().abs() <= 2.0 + 1e-9).all().all()

    def test_with_cot_data(self, prices: pd.DataFrame) -> None:
        sig = FlowSignal()
        rng = np.random.default_rng(0)
        # Weekly COT — use weekly resampled dates
        weekly_dates = prices.resample("W").last().index
        cot = pd.DataFrame(
            rng.normal(0, 1000, (len(weekly_dates), prices.shape[1])),
            index=weekly_dates,
            columns=prices.columns,
        )
        result = sig.compute(prices, cot_net=cot)
        assert result.shape == prices.shape

    def test_with_options_skew(self, prices: pd.DataFrame) -> None:
        sig = FlowSignal()
        skew = pd.DataFrame(
            np.random.default_rng(5).normal(0, 1, prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        result = sig.compute(prices, options_skew=skew)
        assert result.shape == prices.shape

    def test_non_contrarian_mode(self, prices: pd.DataFrame) -> None:
        sig = FlowSignal(contrarian=False)
        result = sig.compute(prices)
        assert result.shape == prices.shape

    def test_contrarian_threshold_zero(self, prices: pd.DataFrame) -> None:
        sig = FlowSignal(contrarian=True, contrarian_threshold=0.0)
        result = sig.compute(prices)
        assert result.shape == prices.shape


class TestComputeCOTSignal:
    """Tests for FlowSignal.compute_cot_signal."""

    def test_output_shape(self, prices: pd.DataFrame) -> None:
        sig = FlowSignal()
        rng = np.random.default_rng(10)
        cot = pd.DataFrame(
            rng.normal(0, 500, prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        result = sig.compute_cot_signal(cot, prices)
        assert result.shape == prices.shape

    def test_clipped_output(self, prices: pd.DataFrame) -> None:
        sig = FlowSignal(z_clip=3.0)
        rng = np.random.default_rng(11)
        cot = pd.DataFrame(
            rng.normal(0, 500, prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        result = sig.compute_cot_signal(cot, prices)
        assert (result.dropna().abs() <= 3.0 + 1e-9).all().all()
