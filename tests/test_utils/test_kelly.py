"""Unit tests for systematic_macro.utils.kelly — 100% coverage."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from systematic_macro.utils.kelly import KellySizer


def _make_cov(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    cov = A @ A.T / n + np.eye(n) * 0.01
    tickers = [f"A{i}" for i in range(n)]
    return pd.DataFrame(cov, index=tickers, columns=tickers)


def _make_mu(n: int, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(n)]
    return pd.Series(rng.uniform(0.001, 0.01, n), index=tickers)


class TestKellySizerInit:
    def test_defaults(self) -> None:
        k = KellySizer()
        assert k.fraction == 0.25
        assert k.max_leverage == 2.0

    def test_invalid_fraction(self) -> None:
        with pytest.raises(ValueError, match="fraction must be in"):
            KellySizer(fraction=0.0)

    def test_invalid_fraction_upper(self) -> None:
        with pytest.raises(ValueError, match="fraction must be in"):
            KellySizer(fraction=1.5)

    def test_invalid_max_leverage(self) -> None:
        with pytest.raises(ValueError, match="max_leverage must be > 0"):
            KellySizer(max_leverage=0.0)

    def test_invalid_max_position(self) -> None:
        with pytest.raises(ValueError, match="max_position must be in"):
            KellySizer(max_position=0.0)


class TestKellyCompute:
    def test_output_is_series(self) -> None:
        k = KellySizer()
        mu = _make_mu(6)
        cov = _make_cov(6)
        w = k.compute(mu, cov)
        assert isinstance(w, pd.Series)

    def test_gross_leverage_bounded(self) -> None:
        k = KellySizer(fraction=0.5, max_leverage=1.5)
        mu = _make_mu(6)
        cov = _make_cov(6)
        w = k.compute(mu, cov)
        assert float(w.abs().sum()) <= 1.5 + 1e-6

    def test_max_position_respected(self) -> None:
        k = KellySizer(max_position=0.15)
        mu = _make_mu(6)
        cov = _make_cov(6)
        w = k.compute(mu, cov)
        assert (w.abs() <= 0.15 + 1e-9).all()

    def test_misaligned_index_raises(self) -> None:
        k = KellySizer()
        mu = pd.Series([0.01, 0.02], index=["X", "Y"])
        cov = _make_cov(2)  # index is A0, A1
        with pytest.raises(ValueError, match="must match"):
            k.compute(mu, cov)

    def test_min_edge_filters_assets(self) -> None:
        k = KellySizer(min_edge=1.0)  # very high threshold
        mu = _make_mu(6)  # all values < 0.01 → filtered
        cov = _make_cov(6)
        w = k.compute(mu, cov)
        assert w.empty

    def test_diagonal_fallback(self) -> None:
        k = KellySizer(use_precision=False)
        mu = _make_mu(4)
        cov = _make_cov(4)
        w = k.compute(mu, cov)
        assert isinstance(w, pd.Series)

    def test_singular_cov_fallback(self) -> None:
        k = KellySizer(use_precision=True)
        # Near-singular cov (rank-deficient)
        tickers = ["A", "B", "C"]
        cov_np = np.ones((3, 3)) * 0.01  # all same → near-singular
        cov = pd.DataFrame(cov_np, index=tickers, columns=tickers)
        mu = pd.Series([0.01, 0.02, 0.005], index=tickers)
        w = k.compute(mu, cov)
        assert isinstance(w, pd.Series)


class TestComputeFromSignal:
    def test_output_shape(self, returns: pd.DataFrame) -> None:
        from systematic_macro.utils.covariance import CovarianceEstimator
        k = KellySizer()
        est = CovarianceEstimator(annualise=True, min_obs=30)
        cov = est.fit(returns)
        signal = pd.Series(1.0, index=returns.columns)
        vol = returns.std() * np.sqrt(252)
        w = k.compute_from_signal(signal, vol, cov, ic_estimate=0.05)
        assert isinstance(w, pd.Series)

    def test_invalid_ic_raises(self, returns: pd.DataFrame) -> None:
        from systematic_macro.utils.covariance import CovarianceEstimator
        k = KellySizer()
        cov = CovarianceEstimator(min_obs=30).fit(returns)
        signal = pd.Series(1.0, index=returns.columns)
        vol = returns.std()
        with pytest.raises(ValueError, match="ic_estimate must be > 0"):
            k.compute_from_signal(signal, vol, cov, ic_estimate=0.0)


class TestDollarSize:
    def test_basic_output_columns(self) -> None:
        k = KellySizer()
        w = pd.Series({"A": 0.5, "B": -0.3, "C": 0.2})
        result = k.dollar_size(w, book_size=100_000_000)
        assert "weight" in result.columns
        assert "notional_usd" in result.columns

    def test_notional_scales_with_book(self) -> None:
        k = KellySizer()
        w = pd.Series({"A": 0.1})
        r1 = k.dollar_size(w, book_size=1_000_000)
        r2 = k.dollar_size(w, book_size=10_000_000)
        assert abs(r2["notional_usd"].iloc[0] / r1["notional_usd"].iloc[0] - 10) < 1e-9

    def test_with_prices_adds_units(self) -> None:
        k = KellySizer()
        w = pd.Series({"SPY": 0.5, "EWJ": -0.2})
        prices = pd.Series({"SPY": 500.0, "EWJ": 60.0})
        result = k.dollar_size(w, book_size=1_000_000, asset_prices=prices)
        assert "units_raw" in result.columns

    def test_invalid_book_size_raises(self) -> None:
        k = KellySizer()
        w = pd.Series({"A": 0.1})
        with pytest.raises(ValueError, match="book_size must be > 0"):
            k.dollar_size(w, book_size=0)
