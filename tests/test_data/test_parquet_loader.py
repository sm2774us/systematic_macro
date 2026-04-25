"""Unit tests for systematic_macro.data.parquet_loader — 100% coverage."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from systematic_macro.data.parquet_loader import ParquetLoader


@pytest.fixture(scope="module")
def tmp_parquet(tmp_path_factory: pytest.TempPathFactory, prices: pd.DataFrame) -> Path:
    """Write a small price DataFrame to a temp Parquet file and return the path."""
    loader = ParquetLoader()
    out = tmp_path_factory.mktemp("data") / "prices.parquet"
    loader.save_prices(prices, path=out)
    return out


class TestParquetLoaderInit:
    def test_defaults(self) -> None:
        loader = ParquetLoader()
        assert loader.date_column == "date"
        assert loader.ticker_column == "ticker"
        assert loader.use_streaming is False


class TestSavePrices:
    def test_creates_file(self, tmp_path: Path, prices: pd.DataFrame) -> None:
        loader = ParquetLoader()
        out = tmp_path / "out.parquet"
        result = loader.save_prices(prices, path=out)
        assert result.exists()

    def test_empty_raises(self, tmp_path: Path) -> None:
        loader = ParquetLoader()
        with pytest.raises(ValueError, match="non-empty"):
            loader.save_prices(pd.DataFrame(), path=tmp_path / "empty.parquet")

    def test_creates_parent_dirs(self, tmp_path: Path, prices: pd.DataFrame) -> None:
        loader = ParquetLoader()
        nested = tmp_path / "a" / "b" / "prices.parquet"
        loader.save_prices(prices, path=nested)
        assert nested.exists()


class TestLoadPrices:
    def test_returns_dataframe(self, tmp_parquet: Path) -> None:
        loader = ParquetLoader()
        df = loader.load_prices(tmp_parquet)
        assert isinstance(df, pd.DataFrame)

    def test_shape_matches_original(self, tmp_parquet: Path, prices: pd.DataFrame) -> None:
        loader = ParquetLoader()
        df = loader.load_prices(tmp_parquet)
        assert df.shape == prices.shape

    def test_ticker_filter(self, tmp_parquet: Path, tickers: list[str]) -> None:
        loader = ParquetLoader()
        subset = tickers[:2]
        df = loader.load_prices(tmp_parquet, tickers=subset)
        assert set(df.columns) == set(subset)

    def test_date_filter_start(self, tmp_parquet: Path, prices: pd.DataFrame) -> None:
        loader = ParquetLoader()
        mid = str(prices.index[len(prices) // 2].date())
        df = loader.load_prices(tmp_parquet, start_date=mid)
        assert len(df) < len(prices)

    def test_date_filter_end(self, tmp_parquet: Path, prices: pd.DataFrame) -> None:
        loader = ParquetLoader()
        mid = str(prices.index[len(prices) // 2].date())
        df = loader.load_prices(tmp_parquet, end_date=mid)
        assert len(df) < len(prices)

    def test_file_not_found_raises(self) -> None:
        loader = ParquetLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_prices("/nonexistent/path/prices.parquet")

    def test_missing_column_raises(self, tmp_path: Path, prices: pd.DataFrame) -> None:
        import polars as pl
        bad = pl.from_pandas(prices.reset_index()).write_parquet(str(tmp_path / "bad.parquet"))
        loader = ParquetLoader(ticker_column="nonexistent_col")
        with pytest.raises(ValueError, match="Required column"):
            loader.load_prices(tmp_path / "bad.parquet")

    def test_index_is_datetime(self, tmp_parquet: Path) -> None:
        loader = ParquetLoader()
        df = loader.load_prices(tmp_parquet)
        assert isinstance(df.index, pd.DatetimeIndex)


class TestComputeReturnsPolars:
    def test_simple_returns_shape(self, prices: pd.DataFrame) -> None:
        loader = ParquetLoader()
        r = loader.compute_returns_polars(prices, method="simple")
        assert r.shape == (len(prices) - 1, prices.shape[1])

    def test_log_returns_shape(self, prices: pd.DataFrame) -> None:
        loader = ParquetLoader()
        r = loader.compute_returns_polars(prices, method="log")
        assert r.shape == (len(prices) - 1, prices.shape[1])

    def test_invalid_method_raises(self, prices: pd.DataFrame) -> None:
        loader = ParquetLoader()
        with pytest.raises(ValueError, match="method must be"):
            loader.compute_returns_polars(prices, method="diff")  # type: ignore[arg-type]

    def test_returns_no_nan_after_first(self, prices: pd.DataFrame) -> None:
        loader = ParquetLoader()
        r = loader.compute_returns_polars(prices, method="simple")
        assert not r.isna().any().any()


class TestLoadAndComputeIC:
    def test_returns_series(
        self, tmp_parquet: Path, prices: pd.DataFrame
    ) -> None:
        loader = ParquetLoader()
        ic = loader.load_and_compute_ic(tmp_parquet, tmp_parquet, horizon=1)
        assert isinstance(ic, pd.Series)
