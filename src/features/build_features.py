"""Feature builders for the dynamic-pricing / demand-forecasting model.

Each ``add_*`` function takes the long-format base frame (one row per
series-day, time-sorted) and returns it with new feature columns. They are pure
and composable so the pipeline can apply them in sequence.

Three families of features (matching Pipeline 1 in the project spec):

1. Lag features        - past sales values (1, 7, 14, 28 days ago).
2. Rolling statistics  - moving mean / std / max of recent sales.
3. Price features      - price changes, discounts, and price elasticity of demand.

Plus calendar / event features that make the model aware of holidays, weekends
and SNAP (food-stamp) days, which strongly shift retail demand.

IMPORTANT (no leakage): every lag and rolling feature is shifted so that the
value for day ``t`` uses only information available *before* day ``t``. Rolling
windows are applied on top of a ``shift(1)`` of sales, so today's sales never
leak into today's own features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

GROUP_KEY = "id"  # one M5 series = unique (item_id, store_id)


# --------------------------------------------------------------------------- #
# 1. Lag features
# --------------------------------------------------------------------------- #
def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add ``sales_lag_{n}`` = units sold n days ago, per series."""
    grouped = df.groupby(GROUP_KEY, observed=True)[config.TARGET]
    for lag in lags:
        df[f"sales_lag_{lag}"] = grouped.shift(lag)
    return df


# --------------------------------------------------------------------------- #
# 2. Rolling statistics
# --------------------------------------------------------------------------- #
def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add rolling mean / std / max of sales over recent windows.

    Computed on ``shift(1)`` of sales so the current day is excluded (no leak).
    """
    # shifted sales: "sales as known yesterday and before"
    shifted = df.groupby(GROUP_KEY, observed=True)[config.TARGET].shift(1)

    for window in windows:
        roll = shifted.groupby(df[GROUP_KEY], observed=True).rolling(
            window, min_periods=1
        )
        df[f"sales_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"sales_roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)
        df[f"sales_roll_max_{window}"] = roll.max().reset_index(level=0, drop=True)
    return df


# --------------------------------------------------------------------------- #
# 3. Price features and price elasticity of demand
# --------------------------------------------------------------------------- #
def add_price_features(df: pd.DataFrame, price_window: int) -> pd.DataFrame:
    """Add price momentum, discount and elasticity features.

    - ``price_change`` / ``price_change_pct``: week-over-week price movement.
    - ``price_roll_mean`` / ``price_discount_pct``: how far today's price sits
      below its recent rolling average (a proxy for "is this on promotion?").
    - ``price_elasticity``: %Δdemand / %Δprice over consecutive days. Captures
      how sensitive a product's sales are to a price change, which the pricing
      optimizer needs. Inelastic (~0) products tolerate higher prices; highly
      elastic products sell much more when discounted.
    """
    grp = df.groupby(GROUP_KEY, observed=True)

    # Previous day's price for this series (price usually changes weekly).
    prev_price = grp[config.PRICE_COL].shift(1)
    df["price_change"] = df[config.PRICE_COL] - prev_price
    df["price_change_pct"] = df["price_change"] / prev_price.replace(0, np.nan)

    # Rolling average price -> discount depth vs. the recent norm.
    price_roll = (
        grp[config.PRICE_COL]
        .shift(1)
        .groupby(df[GROUP_KEY], observed=True)
        .rolling(price_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["price_roll_mean"] = price_roll
    df["price_discount_pct"] = (
        (price_roll - df[config.PRICE_COL]) / price_roll.replace(0, np.nan)
    )

    # Price elasticity of demand: %Δsales / %Δprice between consecutive days.
    prev_sales = grp[config.TARGET].shift(1)
    pct_change_sales = (df[config.TARGET] - prev_sales) / prev_sales.replace(0, np.nan)
    pct_change_price = df["price_change_pct"]
    elasticity = pct_change_sales / pct_change_price.replace(0, np.nan)
    # Only meaningful when the price actually moved; otherwise leave as NaN/0.
    df["price_elasticity"] = elasticity.replace([np.inf, -np.inf], np.nan)

    return df


# --------------------------------------------------------------------------- #
# Calendar / event features
# --------------------------------------------------------------------------- #
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived and event/SNAP flags that shift retail demand."""
    date = df[config.DATE_COL]
    df["day_of_week"] = date.dt.dayofweek            # 0 = Monday
    df["day_of_month"] = date.dt.day
    df["week_of_year"] = date.dt.isocalendar().week.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    # Any named event on this day (holiday, sporting, religious, national).
    df["is_event"] = df["event_name_1"].notna().astype("int8")

    # SNAP benefit day for this series' state -> boosts food sales.
    state_to_snap = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    snap_value = np.zeros(len(df), dtype="int8")
    for state, snap_col in state_to_snap.items():
        if snap_col in df.columns:
            mask = (df["state_id"] == state).to_numpy()
            snap_value[mask] = df.loc[mask, snap_col].fillna(0).astype("int8").to_numpy()
    df["snap_active"] = snap_value

    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_all_features(
    df: pd.DataFrame, cfg: config.FeatureConfig | None = None
) -> pd.DataFrame:
    """Apply every feature family in order and (optionally) trim warm-up rows."""
    cfg = cfg or config.DEFAULT_CONFIG

    df = add_calendar_features(df)
    df = add_lag_features(df, cfg.lag_days)
    df = add_rolling_features(df, cfg.rolling_windows)
    df = add_price_features(df, cfg.price_rolling_window)

    if cfg.drop_warmup_rows:
        # The first ``max_window`` days of each series have undefined lags.
        # Drop them so the model never trains on rows full of NaN features.
        # Vectorized: rank each row within its series by date and keep rows
        # whose position is past the longest look-back window.
        day_rank = df.groupby(GROUP_KEY, observed=True).cumcount()
        df = df[day_rank >= cfg.max_window].reset_index(drop=True)

    if cfg.downcast:
        # Lag / rolling features come back as float64 from pandas; shrink them so
        # the full ~60M-row output stays manageable in memory and on disk.
        from .data_loader import _reduce_mem_usage

        df = _reduce_mem_usage(df)

    return df
