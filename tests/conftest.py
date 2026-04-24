"""Shared pytest fixtures for the systematic_macro test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_macro.data.fetcher import MarketDataFetcher


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic NumPy RNG for all tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def fetcher() -> MarketDataFetcher:
    """Shared MarketDataFetcher instance (synthetic source)."""
    return MarketDataFetcher()


@pytest.fixture(scope="session")
def tickers() -> list[str]:
    """Small cross-section of 6 asset tickers."""
    return ["A", "B", "C", "D", "E", "F"]


@pytest.fixture(scope="session")
def prices(fetcher: MarketDataFetcher, tickers: list[str]) -> pd.DataFrame:
    """2200-row synthetic price DataFrame (≈ 8.7 years, covers 2 macro cycles)."""
    return fetcher.fetch_prices(
        tickers,
        start="2015-01-02",
        end="2023-12-31",
        source="synthetic",
        seed=42,
    )


@pytest.fixture(scope="session")
def returns(fetcher: MarketDataFetcher, prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns computed from prices fixture."""
    return fetcher.compute_returns(prices)


@pytest.fixture(scope="session")
def short_prices(fetcher: MarketDataFetcher, tickers: list[str]) -> pd.DataFrame:
    """Shorter price series (500 rows) for edge-case tests."""
    return fetcher.fetch_prices(
        tickers,
        start="2022-01-03",
        end="2023-12-31",
        source="synthetic",
        seed=7,
    )


@pytest.fixture(scope="session")
def short_returns(fetcher: MarketDataFetcher, short_prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns from short_prices."""
    return fetcher.compute_returns(short_prices)


@pytest.fixture(scope="session")
def mock_signal(returns: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic signal z-scores with mild predictive power."""
    noise = pd.DataFrame(
        rng.standard_normal(returns.shape),
        index=returns.index,
        columns=returns.columns,
    )
    # Weak signal = 10% future return + 90% noise
    fwd = returns.shift(-1).fillna(0)
    raw = 0.1 * fwd + 0.9 * noise
    mu = raw.mean(axis=1)
    sigma = raw.std(axis=1).replace(0, 1)
    return raw.sub(mu, axis=0).div(sigma, axis=0).clip(-3, 3)
