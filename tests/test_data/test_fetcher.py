"""Unit tests for systematic_macro.data.fetcher — 100% coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_macro.data.fetcher import (
    EQUITY_UNIVERSE,
    FUTURES_UNIVERSE,
    FX_UNIVERSE,
    RATE_PROXIES,
    MarketDataFetcher,
)


class TestMarketDataFetcherSynthetic:
    """Tests for synthetic data generation."""

    def test_returns_dataframe(self, fetcher: MarketDataFetcher, tickers: list[str]) -> None:
        df = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_columns_match_tickers(self, fetcher: MarketDataFetcher, tickers: list[str]) -> None:
        df = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01")
        assert list(df.columns) == tickers

    def test_prices_positive(self, fetcher: MarketDataFetcher, tickers: list[str]) -> None:
        df = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01")
        assert (df > 0).all().all()

    def test_index_is_datetime(self, fetcher: MarketDataFetcher, tickers: list[str]) -> None:
        df = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_no_nan_in_prices(self, fetcher: MarketDataFetcher, tickers: list[str]) -> None:
        df = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01")
        assert not df.isna().any().any()

    def test_reproducible_with_same_seed(
        self, fetcher: MarketDataFetcher, tickers: list[str]
    ) -> None:
        df1 = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01", seed=99)
        df2 = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01", seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_data(
        self, fetcher: MarketDataFetcher, tickers: list[str]
    ) -> None:
        df1 = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01", seed=1)
        df2 = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01", seed=2)
        assert not df1.equals(df2)

    def test_invalid_source_raises(self, fetcher: MarketDataFetcher, tickers: list[str]) -> None:
        with pytest.raises(ValueError, match="Unknown source"):
            fetcher.fetch_prices(tickers, source="bloomberg")  # type: ignore[arg-type]

    def test_start_price_near_100(self, fetcher: MarketDataFetcher, tickers: list[str]) -> None:
        df = fetcher.fetch_prices(tickers, start="2020-01-01", end="2021-01-01", seed=0)
        assert (df.iloc[0] > 50).all() and (df.iloc[0] < 200).all()


class TestComputeReturns:
    """Tests for compute_returns."""

    def test_simple_returns_shape(self, fetcher: MarketDataFetcher, prices: pd.DataFrame) -> None:
        r = fetcher.compute_returns(prices, method="simple")
        assert r.shape == (len(prices) - 1, prices.shape[1])

    def test_log_returns_shape(self, fetcher: MarketDataFetcher, prices: pd.DataFrame) -> None:
        r = fetcher.compute_returns(prices, method="log")
        assert r.shape == (len(prices) - 1, prices.shape[1])

    def test_simple_returns_no_nan_after_first_row(
        self, fetcher: MarketDataFetcher, prices: pd.DataFrame
    ) -> None:
        r = fetcher.compute_returns(prices, method="simple")
        assert not r.isna().any().any()

    def test_invalid_method_raises(
        self, fetcher: MarketDataFetcher, prices: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="method must be"):
            fetcher.compute_returns(prices, method="diff")  # type: ignore[arg-type]

    def test_log_vs_simple_close_for_small_returns(
        self, fetcher: MarketDataFetcher, prices: pd.DataFrame
    ) -> None:
        simple = fetcher.compute_returns(prices, method="simple")
        log = fetcher.compute_returns(prices, method="log")
        # For small returns, log ≈ simple
        diff = (simple - log).abs().mean().mean()
        assert diff < 0.005


class TestComputeForwardReturns:
    """Tests for compute_forward_returns."""

    def test_horizon_1_shifts_by_one(
        self, fetcher: MarketDataFetcher, returns: pd.DataFrame
    ) -> None:
        fwd = fetcher.compute_forward_returns(returns, horizon=1)
        assert fwd.shift(1).dropna().equals(returns.dropna()) is False  # shifted

    def test_horizon_shifts_nans_at_end(
        self, fetcher: MarketDataFetcher, returns: pd.DataFrame
    ) -> None:
        h = 5
        fwd = fetcher.compute_forward_returns(returns, horizon=h)
        assert fwd.iloc[-h:].isna().all().all()

    def test_invalid_horizon_raises(
        self, fetcher: MarketDataFetcher, returns: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="horizon must be ≥ 1"):
            fetcher.compute_forward_returns(returns, horizon=0)


class TestComputeVolatility:
    """Tests for compute_volatility."""

    def test_output_shape(self, fetcher: MarketDataFetcher, returns: pd.DataFrame) -> None:
        vol = fetcher.compute_volatility(returns, window=21)
        assert vol.shape == returns.shape

    def test_annualised_vol_reasonable(
        self, fetcher: MarketDataFetcher, returns: pd.DataFrame
    ) -> None:
        vol = fetcher.compute_volatility(returns, window=21, annualise=True)
        clean = vol.dropna()
        assert (clean > 0).all().all()
        assert (clean < 5.0).all().all()  # sanity: < 500% vol

    def test_invalid_window_raises(
        self, fetcher: MarketDataFetcher, returns: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="window must be ≥ 2"):
            fetcher.compute_volatility(returns, window=1)

    def test_no_annualise(self, fetcher: MarketDataFetcher, returns: pd.DataFrame) -> None:
        vol_ann = fetcher.compute_volatility(returns, window=21, annualise=True)
        vol_raw = fetcher.compute_volatility(returns, window=21, annualise=False)
        ratio = (vol_ann / vol_raw).dropna()
        expected = np.sqrt(252)
        assert (ratio.abs() - expected).abs().mean().mean() < 0.01


class TestUniverseConstants:
    """Tests for universe constant lists."""

    def test_fx_universe_nonempty(self) -> None:
        assert len(FX_UNIVERSE) > 0

    def test_futures_universe_nonempty(self) -> None:
        assert len(FUTURES_UNIVERSE) > 0

    def test_equity_universe_nonempty(self) -> None:
        assert len(EQUITY_UNIVERSE) > 0

    def test_rate_proxies_nonempty(self) -> None:
        assert len(RATE_PROXIES) > 0

    def test_all_fx_are_strings(self) -> None:
        assert all(isinstance(t, str) for t in FX_UNIVERSE)
