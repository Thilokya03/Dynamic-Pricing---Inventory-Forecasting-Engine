# tests/test_baseline_model.py

from pathlib import Path
import re
import numpy as np
import pandas as pd
import polars as pl
import pytest

# Change this import according to your project structure.
# Example: from src.models import baseline_model as bm
from scripts import baseline_model as bm


# ------------------------------------------------------------------
# load_daily_sales() tests
# ------------------------------------------------------------------

def test_load_daily_sales_file_not_found(
    monkeypatch,
    tmp_path
):
    missing_file = tmp_path / "missing_sales.parquet"

    monkeypatch.setattr(
        bm,
        "SALES_FILE_PATH",
        missing_file
    )

    expected_message = (
        f"Sales file was not found: {missing_file}"
    )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(expected_message)
    ):
        bm.load_daily_sales()

def test_load_daily_sales_returns_pandas_series(
    monkeypatch,
    tmp_path
):
    """It should return daily sales as a pandas Series."""

    parquet_file = tmp_path / "sales.parquet"

    test_data = pl.DataFrame(
        {
            "date": [
                "2016-04-28",
                "2016-04-28",
                "2016-04-29"
            ],
            "sales": [10, 5, 7]
        }
    ).with_columns(
        pl.col("date").str.to_date()
    )

    test_data.write_parquet(parquet_file)

    monkeypatch.setattr(
        bm,
        "SALES_FILE_PATH",
        parquet_file
    )

    result = bm.load_daily_sales()

    assert isinstance(result, pd.Series)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.name == "total_daily_sales"


def test_load_daily_sales_aggregates_sales_by_date(
    monkeypatch,
    tmp_path
):
    """Sales occurring on the same date should be added together."""

    parquet_file = tmp_path / "sales.parquet"

    test_data = pl.DataFrame(
        {
            "date": [
                "2016-04-28",
                "2016-04-28",
                "2016-04-29",
                "2016-04-29"
            ],
            "sales": [10, 5, 7, 3]
        }
    ).with_columns(
        pl.col("date").str.to_date()
    )

    test_data.write_parquet(parquet_file)

    monkeypatch.setattr(
        bm,
        "SALES_FILE_PATH",
        parquet_file
    )

    result = bm.load_daily_sales()

    assert result.loc[pd.Timestamp("2016-04-28")] == pytest.approx(15.0)
    assert result.loc[pd.Timestamp("2016-04-29")] == pytest.approx(10.0)


def test_load_daily_sales_sorts_dates(
    monkeypatch,
    tmp_path
):
    """The returned time series should be sorted by date."""

    parquet_file = tmp_path / "sales.parquet"

    test_data = pl.DataFrame(
        {
            "date": [
                "2016-05-02",
                "2016-04-30",
                "2016-05-01"
            ],
            "sales": [30, 10, 20]
        }
    ).with_columns(
        pl.col("date").str.to_date()
    )

    test_data.write_parquet(parquet_file)

    monkeypatch.setattr(
        bm,
        "SALES_FILE_PATH",
        parquet_file
    )

    result = bm.load_daily_sales()

    assert result.index.is_monotonic_increasing


def test_load_daily_sales_fills_missing_dates_with_zero(
    monkeypatch,
    tmp_path
):
    """Missing calendar dates should be inserted with zero sales."""

    parquet_file = tmp_path / "sales.parquet"

    test_data = pl.DataFrame(
        {
            "date": [
                "2016-04-28",
                "2016-04-30"
            ],
            "sales": [10, 30]
        }
    ).with_columns(
        pl.col("date").str.to_date()
    )

    test_data.write_parquet(parquet_file)

    monkeypatch.setattr(
        bm,
        "SALES_FILE_PATH",
        parquet_file
    )

    result = bm.load_daily_sales()

    assert len(result) == 3
    assert pd.Timestamp("2016-04-29") in result.index
    assert result.loc[pd.Timestamp("2016-04-29")] == pytest.approx(0.0)


def test_load_daily_sales_returns_float_values(
    monkeypatch,
    tmp_path
):
    """The resulting sales values should use a numeric float dtype."""

    parquet_file = tmp_path / "sales.parquet"

    test_data = pl.DataFrame(
        {
            "date": ["2016-04-28", "2016-04-29"],
            "sales": [10, 20]
        }
    ).with_columns(
        pl.col("date").str.to_date()
    )

    test_data.write_parquet(parquet_file)

    monkeypatch.setattr(
        bm,
        "SALES_FILE_PATH",
        parquet_file
    )

    result = bm.load_daily_sales()

    assert pd.api.types.is_float_dtype(result.dtype)


# ------------------------------------------------------------------
# Fake SARIMAX objects
# ------------------------------------------------------------------

class FakeFittedModel:
    """A lightweight replacement for a fitted SARIMAX model."""

    def __init__(self, forecast_values):
        self.forecast_values = forecast_values

    def forecast(self, steps):
        values = self.forecast_values[:steps]

        return pd.Series(
            values,
            dtype=float
        )

    def summary(self):
        return "Fake SARIMAX summary"

    def save(self, path):
        Path(path).touch()


class FakeSARIMAX:
    """A successful fake SARIMAX implementation."""

    forecast_values = []

    def __init__(
        self,
        train,
        order,
        seasonal_order,
        enforce_stationarity,
        enforce_invertibility
    ):
        self.train = train
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

    def fit(self, disp=False):
        return FakeFittedModel(
            self.forecast_values
        )


class FailingSARIMAX:
    """A fake SARIMAX implementation that always fails."""

    def __init__(self, *args, **kwargs):
        raise ValueError("Model fitting failed")


# ------------------------------------------------------------------
# get_baseline_model() tests
# ------------------------------------------------------------------

@pytest.fixture
def sample_daily_sales():
    """Create train and test data around the configured split date."""

    dates = pd.date_range(
        start="2016-04-25",
        end="2016-05-04",
        freq="D"
    )

    values = [
        10,
        12,
        14,
        16,
        18,
        20,  # 2016-04-30: last training observation
        22,
        24,
        26,
        28
    ]

    return pd.Series(
        values,
        index=dates,
        name="total_daily_sales",
        dtype=float
    )


def test_get_baseline_model_returns_expected_fields(
    monkeypatch,
    sample_daily_sales
):
    """The function should return all information about the best model."""

    monkeypatch.setattr(
        bm,
        "load_daily_sales",
        lambda: sample_daily_sales
    )

    FakeSARIMAX.forecast_values = [
        22,
        24,
        26,
        28
    ]

    monkeypatch.setattr(
        bm,
        "SARIMAX",
        FakeSARIMAX
    )

    monkeypatch.setattr(
        bm,
        "CANDIDATES",
        [((1, 0, 0), (0, 1, 1, 7))]
    )

    result = bm.get_baseline_model()

    assert isinstance(result, dict)

    assert set(result.keys()) == {
        "model",
        "rmse",
        "mae",
        "order",
        "seasonal_order",
        "forecast"
    }

    assert isinstance(result["model"], FakeFittedModel)
    assert isinstance(result["forecast"], pd.Series)


def test_get_baseline_model_calculates_zero_error_for_perfect_forecast(
    monkeypatch,
    sample_daily_sales
):
    """RMSE and MAE should be zero for a perfect forecast."""

    monkeypatch.setattr(
        bm,
        "load_daily_sales",
        lambda: sample_daily_sales
    )

    FakeSARIMAX.forecast_values = [
        22,
        24,
        26,
        28
    ]

    monkeypatch.setattr(
        bm,
        "SARIMAX",
        FakeSARIMAX
    )

    monkeypatch.setattr(
        bm,
        "CANDIDATES",
        [((1, 0, 0), (0, 1, 1, 7))]
    )

    result = bm.get_baseline_model()

    assert result["rmse"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(0.0)


def test_get_baseline_model_returns_candidate_orders(
    monkeypatch,
    sample_daily_sales
):
    """The selected SARIMAX orders should be included in the result."""

    expected_order = (1, 0, 0)
    expected_seasonal_order = (0, 1, 1, 7)

    monkeypatch.setattr(
        bm,
        "load_daily_sales",
        lambda: sample_daily_sales
    )

    FakeSARIMAX.forecast_values = [
        22,
        24,
        26,
        28
    ]

    monkeypatch.setattr(
        bm,
        "SARIMAX",
        FakeSARIMAX
    )

    monkeypatch.setattr(
        bm,
        "CANDIDATES",
        [
            (
                expected_order,
                expected_seasonal_order
            )
        ]
    )

    result = bm.get_baseline_model()

    assert result["order"] == expected_order
    assert result["seasonal_order"] == expected_seasonal_order


def test_get_baseline_model_raises_for_empty_training_data(
    monkeypatch
):
    """The function should fail when there is no data before the split."""

    test_only_series = pd.Series(
        [10, 20, 30],
        index=pd.date_range(
            start="2016-05-01",
            periods=3,
            freq="D"
        ),
        dtype=float
    )

    monkeypatch.setattr(
        bm,
        "load_daily_sales",
        lambda: test_only_series
    )

    with pytest.raises(
        ValueError,
        match="Training dataset is empty"
    ):
        bm.get_baseline_model()


def test_get_baseline_model_raises_for_empty_test_data(
    monkeypatch
):
    """The function should fail when there is no data after the split."""

    train_only_series = pd.Series(
        [10, 20, 30],
        index=pd.date_range(
            start="2016-04-20",
            periods=3,
            freq="D"
        ),
        dtype=float
    )

    monkeypatch.setattr(
        bm,
        "load_daily_sales",
        lambda: train_only_series
    )

    with pytest.raises(
        ValueError,
        match="Testing dataset is empty"
    ):
        bm.get_baseline_model()


def test_get_baseline_model_raises_when_all_candidates_fail(
    monkeypatch,
    sample_daily_sales
):
    """A RuntimeError should be raised if no candidate can be fitted."""

    monkeypatch.setattr(
        bm,
        "load_daily_sales",
        lambda: sample_daily_sales
    )

    monkeypatch.setattr(
        bm,
        "SARIMAX",
        FailingSARIMAX
    )

    monkeypatch.setattr(
        bm,
        "CANDIDATES",
        [
            ((1, 0, 0), (0, 1, 1, 7)),
            ((0, 0, 1), (0, 1, 1, 7))
        ]
    )

    with pytest.warns(
        UserWarning,
        match="Model .* failed"
    ):
        with pytest.raises(
            RuntimeError,
            match="All SARIMAX candidate models failed"
        ):
            bm.get_baseline_model()