"""Load the raw M5 files and reshape them into a single tidy long-format frame.

The raw sales file is "wide": one row per (item, store) series with one column
per day (``d_1`` ... ``d_1941``). Machine learning needs a "long" frame: one row
per (series, day). This module performs that reshape and joins on the calendar
(to get real dates / events) and sell_prices (to get the price for each row).

Output schema (one row per series-day)::

    id, item_id, dept_id, cat_id, store_id, state_id,
    d, sales, date, wm_yr_wk, weekday, wday, month, year,
    event_name_1, event_type_1, event_name_2, event_type_2,
    snap_CA, snap_TX, snap_WI, sell_price
"""

from __future__ import annotations

import pandas as pd

from . import config


def _reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to the smallest dtype that fits the values."""
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float", "float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_calendar() -> pd.DataFrame:
    """Read calendar.csv with proper dtypes and a parsed ``date`` column."""
    cal = pd.read_csv(config.CALENDAR_CSV, parse_dates=[config.DATE_COL])
    # SNAP flags and wday/month/year are small integers.
    int_cols = ["wm_yr_wk", "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"]
    for col in int_cols:
        cal[col] = pd.to_numeric(cal[col], downcast="integer")
    return cal


def load_sell_prices() -> pd.DataFrame:
    """Read sell_prices.csv (store_id, item_id, wm_yr_wk, sell_price)."""
    prices = pd.read_csv(config.SELL_PRICES_CSV)
    prices["wm_yr_wk"] = pd.to_numeric(prices["wm_yr_wk"], downcast="integer")
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], downcast="float")
    return prices


def _sales_path() -> str:
    """Prefer the more complete evaluation file; fall back to validation."""
    if config.SALES_EVALUATION_CSV.exists():
        return str(config.SALES_EVALUATION_CSV)
    if config.SALES_VALIDATION_CSV.exists():
        return str(config.SALES_VALIDATION_CSV)
    raise FileNotFoundError(
        "No sales file found. Expected one of:\n"
        f"  {config.SALES_EVALUATION_CSV}\n"
        f"  {config.SALES_VALIDATION_CSV}"
    )


def load_sales_long(sample_items: int | None = None) -> pd.DataFrame:
    """Load the wide sales file and melt it into long (series-day) format.

    Parameters
    ----------
    sample_items:
        If given, keep only the first N unique ``item_id`` values. This makes
        local smoke tests fast (the full melt is ~60M rows).
    """
    path = _sales_path()
    wide = pd.read_csv(path)

    if sample_items is not None:
        keep = wide["item_id"].drop_duplicates().head(sample_items)
        wide = wide[wide["item_id"].isin(keep)].copy()

    day_cols = [c for c in wide.columns if c.startswith("d_")]

    long_df = wide.melt(
        id_vars=config.ID_COLS,
        value_vars=day_cols,
        var_name=config.DAY_COL,
        value_name=config.TARGET,
    )
    long_df[config.TARGET] = pd.to_numeric(long_df[config.TARGET], downcast="integer")
    return long_df


def build_base_frame(cfg: config.FeatureConfig | None = None) -> pd.DataFrame:
    """Produce the merged long-format base frame ready for feature building.

    Joins: melted sales  +  calendar (on ``d``)  +  sell_prices (on
    store_id/item_id/wm_yr_wk). Rows are sorted by series and date so that the
    lag / rolling features computed downstream are time-ordered.
    """
    cfg = cfg or config.DEFAULT_CONFIG

    sales = load_sales_long(sample_items=cfg.sample_items)
    calendar = load_calendar()
    prices = load_sell_prices()

    # Attach real calendar info (date, events, snap, week id) to each series-day.
    calendar_cols = [
        config.DAY_COL, config.DATE_COL, config.WEEK_COL,
        "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    df = sales.merge(calendar[calendar_cols], on=config.DAY_COL, how="left")

    # Attach the sell price for the (store, item, week) of each row.
    df = df.merge(
        prices,
        on=["store_id", "item_id", config.WEEK_COL],
        how="left",
    )

    # Time order is essential for correct lag/rolling computation.
    df = df.sort_values(["id", config.DATE_COL]).reset_index(drop=True)

    if cfg.downcast:
        df = _reduce_mem_usage(df)

    return df
