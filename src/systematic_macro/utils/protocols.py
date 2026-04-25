# Copyright 2026 Systematic Macro Research. All rights reserved.
"""Structural typing Protocols for tradeable assets across asset classes.

Defines a ``TradeableAsset`` Protocol that enforces a common interface
across FX spots, futures contracts, and equity ETFs — enabling generic
signal and sizing code that operates correctly regardless of the asset
class-specific decimal precision or contract specifications.

Typical usage::

    from systematic_macro.utils.protocols import TradeableAsset, FXSpot
    def size_trade(asset: TradeableAsset, notional: float) -> float:
        return round(notional / asset.lot_size, asset.decimal_places)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class TradeableAsset(Protocol):
    """Structural protocol for any asset the system can trade.

    All concrete asset classes must implement these attributes/methods
    to participate in sizing, risk, and execution logic.
    """

    @property
    def ticker(self) -> str:
        """Bloomberg/exchange ticker symbol."""
        ...

    @property
    def asset_class(self) -> str:
        """One of ``'fx'``, ``'futures'``, ``'equity'``."""
        ...

    @property
    def decimal_places(self) -> int:
        """Price decimal precision (e.g. 5 for FX, 2 for equities)."""
        ...

    @property
    def lot_size(self) -> float:
        """Minimum tradeable unit (1 for equities, 100_000 for FX majors)."""
        ...

    @property
    def point_value(self) -> float:
        """P&L per 1-unit move (1.0 for most; 50.0 for ES futures)."""
        ...

    def notional(self, price: float, quantity: float) -> float:
        """Compute dollar notional for ``quantity`` units at ``price``."""
        ...


@dataclass(frozen=True)
class FXSpot:
    """Concrete FX spot asset implementing :class:`TradeableAsset`.

    Attributes:
        ticker: Yahoo/broker ticker (e.g. ``"EURUSD=X"``).
        decimal_places: Price precision (5 for majors, 3 for JPY pairs).
        lot_size: Standard lot size in base currency units.
        point_value: P&L per pip (lot_size × pip size).
    """

    ticker: str
    decimal_places: int = 5
    lot_size: float = 100_000.0
    point_value: float = 10.0  # USD per pip per standard lot

    @property
    def asset_class(self) -> str:
        """Return asset class label."""
        return "fx"

    def notional(self, price: float, quantity: float) -> float:
        """USD notional = quantity lots × lot_size × price.

        Args:
            price: Current spot price.
            quantity: Number of standard lots (may be fractional).

        Returns:
            Dollar notional exposure.

        Raises:
            ValueError: If ``price`` ≤ 0.
        """
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        return quantity * self.lot_size * price


@dataclass(frozen=True)
class FuturesContract:
    """Concrete futures contract implementing :class:`TradeableAsset`.

    Attributes:
        ticker: Exchange ticker (e.g. ``"ES=F"`` for S&P 500 E-mini).
        decimal_places: Price decimal precision.
        lot_size: Number of contracts per trade unit (usually 1).
        point_value: Dollar value per index point (50 for ES, 10 for NQ).
    """

    ticker: str
    decimal_places: int = 2
    lot_size: float = 1.0
    point_value: float = 50.0  # S&P E-mini default

    @property
    def asset_class(self) -> str:
        """Return asset class label."""
        return "futures"

    def notional(self, price: float, quantity: float) -> float:
        """Dollar notional = quantity × price × point_value.

        Args:
            price: Futures price in index points.
            quantity: Number of contracts.

        Returns:
            Dollar notional exposure.

        Raises:
            ValueError: If ``price`` ≤ 0.
        """
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        return quantity * price * self.point_value


@dataclass(frozen=True)
class EquityETF:
    """Concrete equity ETF implementing :class:`TradeableAsset`.

    Attributes:
        ticker: Exchange ticker (e.g. ``"SPY"``).
        decimal_places: Price precision (2 for USD-denominated ETFs).
        lot_size: Minimum order size (1 share for ETFs).
        point_value: Always 1.0 for equity instruments.
    """

    ticker: str
    decimal_places: int = 2
    lot_size: float = 1.0
    point_value: float = 1.0

    @property
    def asset_class(self) -> str:
        """Return asset class label."""
        return "equity"

    def notional(self, price: float, quantity: float) -> float:
        """Dollar notional = quantity shares × price.

        Args:
            price: Current share price.
            quantity: Number of shares.

        Returns:
            Dollar notional exposure.

        Raises:
            ValueError: If ``price`` ≤ 0.
        """
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        return quantity * price
