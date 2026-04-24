"""Market data fetching, preprocessing, and universe definitions."""

from systematic_macro.data.fetcher import (
    EQUITY_UNIVERSE,
    FUTURES_UNIVERSE,
    FX_UNIVERSE,
    RATE_PROXIES,
    MarketDataFetcher,
)

__all__ = [
    "MarketDataFetcher",
    "FX_UNIVERSE",
    "FUTURES_UNIVERSE",
    "EQUITY_UNIVERSE",
    "RATE_PROXIES",
]
