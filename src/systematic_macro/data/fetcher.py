# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Market data fetching and preprocessing for FX, Futures, and Equities.

Supports yfinance (live/historical) and synthetic data generation for testing.
All returned DataFrames are indexed by ``pd.DatetimeIndex`` with timezone-naive
UTC dates, columns are asset tickers.

Typical usage::

    from systematic_macro.data.fetcher import MarketDataFetcher
    fetcher = MarketDataFetcher()
    prices = fetcher.fetch_prices(["EURUSD=X", "GC=F"], start="2020-01-01")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Universe definitions — 2026 macro environment
# ---------------------------------------------------------------------------

#: G10 FX tickers (vs USD on Yahoo Finance)
FX_UNIVERSE: list[str] = [
    "EURUSD=X",  # EUR/USD
    "GBPUSD=X",  # GBP/USD
    "AUDUSD=X",  # AUD/USD
    "NZDUSD=X",  # NZD/USD
    "USDCAD=X",  # USD/CAD  (quote inverted — handled internally)
    "USDJPY=X",  # USD/JPY  (quote inverted)
    "USDCHF=X",  # USD/CHF  (quote inverted)
    "USDSEK=X",  # USD/SEK  (quote inverted)
    "USDNOK=X",  # USD/NOK  (quote inverted)
]

#: Commodity futures tickers
FUTURES_UNIVERSE: list[str] = [
    "GC=F",   # Gold
    "CL=F",   # WTI Crude
    "SI=F",   # Silver
    "HG=F",   # Copper
    "NG=F",   # Natural Gas
    "ZN=F",   # 10Y US Treasury
    "ZB=F",   # 30Y US Treasury
    "6E=F",   # EUR FX Futures
    "ES=F",   # S&P 500 E-mini
]

#: Equity ETF proxies for cross-country momentum
EQUITY_UNIVERSE: list[str] = [
    "SPY",   # US Large Cap
    "EWJ",   # Japan
    "EWG",   # Germany
    "EWU",   # UK
    "EWA",   # Australia
    "EWC",   # Canada
    "MCHI",  # China
    "EEM",   # EM Broad
    "EWZ",   # Brazil
    "INDA",  # India
]

#: Short-term interest rate proxies for carry (3M rates via ETF/index)
RATE_PROXIES: dict[str, str] = {
    "USD": "^IRX",    # 13-week T-bill
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "JPY": "USDJPY=X",
    "AUD": "AUDUSD=X",
}


@dataclass
class MarketDataFetcher:
    """Fetches, caches, and pre-processes market data.

    Attributes:
        cache: In-memory price cache (ticker -> DataFrame).
        adjusted: Whether to use adjusted close prices (splits + dividends).
    """

    cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    adjusted: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_prices(
        self,
        tickers: list[str],
        start: str = "2015-01-01",
        end: str | None = None,
        source: Literal["yfinance", "synthetic"] = "synthetic",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Fetch daily close prices for a list of tickers.

        Args:
            tickers: List of ticker symbols.
            start: Start date string (``"YYYY-MM-DD"``).
            end: End date string; defaults to today.
            source: ``"yfinance"`` for live data or ``"synthetic"`` for
                reproducible test data.
            seed: Random seed (only used when ``source="synthetic"``).

        Returns:
            (T × N) DataFrame of prices, DatetimeIndex, columns=tickers.
            Missing values forward-filled then back-filled.

        Raises:
            ValueError: If ``source`` is not recognised.
        """
        if source == "synthetic":
            return self._generate_synthetic(tickers, start, end, seed=seed)
        elif source == "yfinance":
            return self._fetch_yfinance(tickers, start, end)
        else:
            raise ValueError(f"Unknown source: {source!r}")

    def compute_returns(
        self,
        prices: pd.DataFrame,
        method: Literal["simple", "log"] = "simple",
    ) -> pd.DataFrame:
        """Compute period returns from a price DataFrame.

        Args:
            prices: (T × N) price DataFrame.
            method: ``"simple"`` (arithmetic) or ``"log"`` (continuous).

        Returns:
            (T-1 × N) returns DataFrame (first row dropped).

        Raises:
            ValueError: If ``method`` is not ``"simple"`` or ``"log"``.
        """
        if method == "simple":
            return prices.pct_change().iloc[1:]
        elif method == "log":
            return np.log(prices / prices.shift(1)).iloc[1:]
        else:
            raise ValueError(f"method must be 'simple' or 'log', got {method!r}")

    def compute_forward_returns(
        self,
        returns: pd.DataFrame,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Shift returns to align signal date with *future* return.

        Args:
            returns: (T × N) returns DataFrame.
            horizon: Forward periods (e.g. 1 = next-day return, 5 = weekly).

        Returns:
            Forward-shifted returns; last ``horizon`` rows are NaN.

        Raises:
            ValueError: If ``horizon`` < 1.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be ≥ 1, got {horizon}")
        return returns.shift(-horizon)

    def compute_volatility(
        self,
        returns: pd.DataFrame,
        window: int = 21,
        annualise: bool = True,
        freq: int = 252,
    ) -> pd.DataFrame:
        """Rolling realised volatility.

        Args:
            returns: (T × N) period returns.
            window: Rolling window in periods.
            annualise: If ``True``, multiply by ``sqrt(freq)``.
            freq: Trading days per year.

        Returns:
            (T × N) volatility DataFrame.

        Raises:
            ValueError: If ``window`` < 2.
        """
        if window < 2:
            raise ValueError(f"window must be ≥ 2, got {window}")
        vol = returns.rolling(window).std(ddof=1)
        if annualise:
            vol = vol * np.sqrt(freq)
        return vol

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_yfinance(
        self,
        tickers: list[str],
        start: str,
        end: str | None,
    ) -> pd.DataFrame:
        """Fetch prices via yfinance with caching and robust error handling."""
        try:
            import yfinance as yf  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("yfinance not installed; use source='synthetic'") from exc

        key = f"{','.join(sorted(tickers))}|{start}|{end}"
        if key in self.cache:
            logger.debug("Cache hit for {}", key)
            return self.cache[key]

        logger.info("Fetching {} tickers from yfinance ({} → {})", len(tickers), start, end)
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=self.adjusted,
            progress=False,
            threads=True,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw

        prices = (
            prices
            .ffill()
            .bfill()
            .dropna(how="all")
        )
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        self.cache[key] = prices
        return prices

    @staticmethod
    def _generate_synthetic(
        tickers: list[str],
        start: str,
        end: str | None,
        seed: int = 42,
        annual_vol: float = 0.15,
        annual_drift: float = 0.05,
    ) -> pd.DataFrame:
        """Generate synthetic GBM prices for reproducible testing.

        Args:
            tickers: List of ticker names (used as column labels).
            start: Start date.
            end: End date (defaults to today).
            seed: NumPy random seed.
            annual_vol: Annualised volatility for GBM.
            annual_drift: Annualised drift for GBM.

        Returns:
            (T × N) DataFrame of synthetic prices starting at 100.
        """
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start=start, end=end or pd.Timestamp.today())
        dt = 1.0 / 252.0
        n = len(dates)
        k = len(tickers)

        drift = (annual_drift - 0.5 * annual_vol**2) * dt
        diffusion = annual_vol * np.sqrt(dt)

        shocks = rng.standard_normal((n, k)) * diffusion + drift
        log_returns = np.vstack([np.zeros((1, k)), shocks])
        prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))[:n]

        return pd.DataFrame(prices, index=dates, columns=tickers)
