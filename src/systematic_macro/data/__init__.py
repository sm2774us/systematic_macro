"""Market data fetching, Parquet loading, preprocessing, and universe definitions."""

from systematic_macro.data.fetcher import (
    EQUITY_UNIVERSE, FUTURES_UNIVERSE, FX_UNIVERSE, RATE_PROXIES, MarketDataFetcher,
)
from systematic_macro.data.parquet_loader import ParquetLoader

__all__ = [
    "MarketDataFetcher", "ParquetLoader",
    "FX_UNIVERSE", "FUTURES_UNIVERSE", "EQUITY_UNIVERSE", "RATE_PROXIES",
]
