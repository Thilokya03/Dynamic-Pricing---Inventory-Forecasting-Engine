import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SALES_FILE_PATH = PROJECT_ROOT / "data" / "silver" / "sales.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE_PATH = MODEL_DIR / "sarimax_baseline.pkl"


CANDIDATES = [
    # order       seasonal_order
    ((1, 0, 0), (0, 1, 1, 7)),
    ((0, 0, 1), (0, 1, 1, 7)),
    ((1, 0, 1), (0, 1, 1, 7)),
    ((1, 0, 0), (1, 1, 0, 7)),
    ((0, 0, 1), (1, 1, 0, 7)),
    ((1, 0, 1), (1, 1, 1, 7)),
]


def load_daily_sales() -> pd.Series:
    if not SALES_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Sales file was not found: {SALES_FILE_PATH}"
        )

    df = pl.read_parquet(
        SALES_FILE_PATH,
        columns=["date", "sales"]
    )

    daily_summary = (
        df
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("sales").cast(pl.Float64)
        )
        .group_by("date")
        .agg(
            pl.sum("sales").alias("total_daily_sales")
        )
        .sort("date")
        .to_pandas()
    )

    daily_summary["date"] = pd.to_datetime(daily_summary["date"])

    daily_sales = (
        daily_summary
        .set_index("date")["total_daily_sales"]
        .sort_index()
        .asfreq("D", fill_value=0)
        .astype(float)
    )

    return daily_sales


def get_baseline_model():
    daily_sales = load_daily_sales()

    split_date = pd.Timestamp("2016-05-01")

    train = daily_sales.loc[daily_sales.index < split_date]
    test = daily_sales.loc[daily_sales.index >= split_date]

    if train.empty:
        raise ValueError("Training dataset is empty.")

    if test.empty:
        raise ValueError("Testing dataset is empty.")

    best_result = None

    for order, seasonal_order in CANDIDATES:
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted_model = model.fit(disp=False)

            forecast = fitted_model.forecast(steps=len(test))
            forecast.index = test.index

            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))

            print(
                f"Order={order}, "
                f"Seasonal order={seasonal_order}, "
                f"RMSE={rmse:.4f}, "
                f"MAE={mae:.4f}"
            )

            if best_result is None or rmse < best_result["rmse"]:
                best_result = {
                    "model": fitted_model,
                    "rmse": rmse,
                    "mae": mae,
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "forecast": forecast
                }

        except Exception as error:
            warnings.warn(
                f"Model {order}, {seasonal_order} failed: {error}"
            )

    if best_result is None:
        raise RuntimeError("All SARIMAX candidate models failed.")

    return best_result


if __name__ == "__main__":
    result = get_baseline_model()

    print("\nBest Model Summary:")
    print(result["model"].summary())

    print(f"\nRMSE: {result['rmse']:.4f}")
    print(f"MAE: {result['mae']:.4f}")
    print(f"Order: {result['order']}")
    print(f"Seasonal Order: {result['seasonal_order']}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result["model"].save(MODEL_FILE_PATH)

    print(f"Model saved successfully to: {MODEL_FILE_PATH}")