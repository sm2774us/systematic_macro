"""Unit tests for systematic_macro.portfolio.optimizer — 100% coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_macro.portfolio.optimizer import PortfolioOptimizer


class TestPortfolioOptimizerInit:
    """Tests for PortfolioOptimizer initialisation and validation."""

    def test_defaults(self) -> None:
        opt = PortfolioOptimizer()
        assert opt.method == "signal_scaled"
        assert opt.target_vol == 0.10
        assert opt.max_position == 0.25

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            PortfolioOptimizer(method="black_litterman")  # type: ignore[arg-type]

    def test_invalid_target_vol_raises(self) -> None:
        with pytest.raises(ValueError, match="target_vol must be in"):
            PortfolioOptimizer(target_vol=1.5)

    def test_invalid_max_position_raises(self) -> None:
        with pytest.raises(ValueError, match="max_position must be in"):
            PortfolioOptimizer(max_position=1.5)


class TestComputeWeights:
    """Tests for PortfolioOptimizer.compute_weights."""

    def test_output_shape_signal_scaled(
        self, mock_signal: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        opt = PortfolioOptimizer(method="signal_scaled")
        w = opt.compute_weights(mock_signal, returns)
        assert w.shape == mock_signal.shape

    def test_output_shape_risk_parity(
        self, mock_signal: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        opt = PortfolioOptimizer(method="risk_parity")
        w = opt.compute_weights(mock_signal.iloc[-200:], returns)
        assert w.shape == mock_signal.iloc[-200:].shape

    def test_output_shape_mean_variance(
        self, mock_signal: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        opt = PortfolioOptimizer(method="mean_variance")
        w = opt.compute_weights(mock_signal.iloc[-200:], returns)
        assert w.shape == mock_signal.iloc[-200:].shape

    def test_weights_bounded_by_max_position(
        self, mock_signal: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        opt = PortfolioOptimizer(max_position=0.20)
        w = opt.compute_weights(mock_signal, returns)
        assert (w.abs() <= 0.20 + 1e-9).all().all()

    def test_mismatched_columns_raises(
        self, mock_signal: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        opt = PortfolioOptimizer()
        bad_signal = mock_signal.copy()
        bad_signal.columns = [f"X{i}" for i in range(bad_signal.shape[1])]
        with pytest.raises(ValueError, match="subset of returns columns"):
            opt.compute_weights(bad_signal, returns)


class TestBlendSignals:
    """Tests for PortfolioOptimizer.blend_signals."""

    def test_equal_weight_blend(self, mock_signal: pd.DataFrame) -> None:
        opt = PortfolioOptimizer()
        signals = {"carry": mock_signal, "momentum": mock_signal * 0.5}
        result = opt.blend_signals(signals)
        assert result.shape == mock_signal.shape

    def test_custom_weights(self, mock_signal: pd.DataFrame) -> None:
        opt = PortfolioOptimizer(signal_weights={"carry": 0.7, "momentum": 0.3})
        signals = {"carry": mock_signal, "momentum": mock_signal * 0.5}
        result = opt.blend_signals(signals)
        assert result.shape == mock_signal.shape

    def test_empty_signals_raises(self) -> None:
        opt = PortfolioOptimizer()
        with pytest.raises(ValueError, match="must not be empty"):
            opt.blend_signals({})

    def test_missing_signal_weight_key_raises(self, mock_signal: pd.DataFrame) -> None:
        opt = PortfolioOptimizer(signal_weights={"carry": 1.0})
        with pytest.raises(ValueError, match="missing keys"):
            opt.blend_signals({"carry": mock_signal, "momentum": mock_signal})

    def test_single_signal_passthrough(self, mock_signal: pd.DataFrame) -> None:
        opt = PortfolioOptimizer()
        result = opt.blend_signals({"carry": mock_signal})
        pd.testing.assert_frame_equal(result, mock_signal)


class TestVolScaleWeights:
    """Tests for PortfolioOptimizer.vol_scale_weights."""

    def test_output_shape(
        self, mock_signal: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        opt = PortfolioOptimizer(target_vol=0.10)
        w = opt.compute_weights(mock_signal, returns)
        scaled = opt.vol_scale_weights(w, returns)
        assert scaled.shape == w.shape

    def test_max_position_still_enforced(
        self, mock_signal: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        opt = PortfolioOptimizer(target_vol=0.10, max_position=0.20)
        w = opt.compute_weights(mock_signal, returns)
        scaled = opt.vol_scale_weights(w, returns)
        assert (scaled.abs() <= 0.20 + 1e-9).all().all()
