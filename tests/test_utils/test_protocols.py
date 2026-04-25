"""Unit tests for systematic_macro.utils.protocols — 100% coverage."""
from __future__ import annotations
import pytest
from systematic_macro.utils.protocols import (
    EquityETF, FuturesContract, FXSpot, TradeableAsset,
)


class TestFXSpot:
    def test_asset_class(self) -> None:
        fx = FXSpot(ticker="EURUSD=X")
        assert fx.asset_class == "fx"

    def test_notional(self) -> None:
        fx = FXSpot(ticker="EURUSD=X", lot_size=100_000)
        notional = fx.notional(price=1.08, quantity=2.0)
        assert abs(notional - 216_000.0) < 1e-6

    def test_invalid_price_raises(self) -> None:
        fx = FXSpot(ticker="EURUSD=X")
        with pytest.raises(ValueError, match="price must be > 0"):
            fx.notional(price=0.0, quantity=1.0)

    def test_is_tradeable_asset(self) -> None:
        fx = FXSpot(ticker="EURUSD=X")
        assert isinstance(fx, TradeableAsset)

    def test_defaults(self) -> None:
        fx = FXSpot(ticker="GBPUSD=X")
        assert fx.decimal_places == 5
        assert fx.lot_size == 100_000.0


class TestFuturesContract:
    def test_asset_class(self) -> None:
        fut = FuturesContract(ticker="ES=F")
        assert fut.asset_class == "futures"

    def test_notional(self) -> None:
        fut = FuturesContract(ticker="ES=F", point_value=50.0)
        notional = fut.notional(price=5000.0, quantity=2.0)
        assert abs(notional - 500_000.0) < 1e-6

    def test_invalid_price_raises(self) -> None:
        fut = FuturesContract(ticker="GC=F")
        with pytest.raises(ValueError, match="price must be > 0"):
            fut.notional(price=-1.0, quantity=1.0)

    def test_is_tradeable_asset(self) -> None:
        assert isinstance(FuturesContract(ticker="CL=F"), TradeableAsset)

    def test_defaults(self) -> None:
        fut = FuturesContract(ticker="NQ=F")
        assert fut.decimal_places == 2
        assert fut.lot_size == 1.0


class TestEquityETF:
    def test_asset_class(self) -> None:
        etf = EquityETF(ticker="SPY")
        assert etf.asset_class == "equity"

    def test_notional(self) -> None:
        etf = EquityETF(ticker="SPY")
        notional = etf.notional(price=500.0, quantity=100.0)
        assert abs(notional - 50_000.0) < 1e-6

    def test_invalid_price_raises(self) -> None:
        etf = EquityETF(ticker="EWJ")
        with pytest.raises(ValueError, match="price must be > 0"):
            etf.notional(price=0.0, quantity=10.0)

    def test_is_tradeable_asset(self) -> None:
        assert isinstance(EquityETF(ticker="EEM"), TradeableAsset)

    def test_point_value_is_one(self) -> None:
        assert EquityETF(ticker="SPY").point_value == 1.0


class TestTradeableAssetProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        fx = FXSpot(ticker="USDJPY=X")
        assert isinstance(fx, TradeableAsset)

    def test_plain_object_not_tradeable(self) -> None:
        class NotAnAsset:
            pass
        assert not isinstance(NotAnAsset(), TradeableAsset)
