"""Central configuration for the feature engineering pipeline.

All paths, column names, and feature parameters live here so the rest of the
pipeline stays declarative and the team can reproduce the exact same dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = parent of the `src` folder (.../<project>/src/features/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Raw M5 files
CALENDAR_CSV = RAW_DATA_DIR / "calendar.csv"
SELL_PRICES_CSV = RAW_DATA_DIR / "sell_prices.csv"
# The "evaluation" file is the most complete (includes the extra 28 days that
# "validation" lacks); fall back to validation if evaluation is missing.
SALES_EVALUATION_CSV = RAW_DATA_DIR / "sales_train_evaluation.csv"
SALES_VALIDATION_CSV = RAW_DATA_DIR / "sales_train_validation.csv"

# Default output of the pipeline
FEATURES_PARQUET = PROCESSED_DATA_DIR / "features.parquet"


# --------------------------------------------------------------------------- #
# Column names (M5 schema)
# --------------------------------------------------------------------------- #
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
TARGET = "sales"          # daily units sold (long format)
DAY_COL = "d"             # e.g. "d_1", "d_2", ...
DATE_COL = "date"
WEEK_COL = "wm_yr_wk"     # Walmart year-week, links sales to sell_prices
PRICE_COL = "sell_price"

# Categorical columns that the downstream model will treat as categories.
CATEGORICAL_COLS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "weekday",
]


# --------------------------------------------------------------------------- #
# Feature parameters
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    """Knobs for the feature engineering pipeline.

    Defaults are sensible for the M5 daily horizon. ``sample_items`` lets you
    run the whole pipeline on a small slice for quick local testing.
    """

    # Lag features: how many days back to look at past sales.
    lag_days: list[int] = field(default_factory=lambda: [1, 7, 14, 28])

    # Rolling-window sizes (in days) for mean / std / max statistics.
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14, 28])

    # Price rolling window used to derive price momentum / discount features.
    price_rolling_window: int = 28

    # Drop the warm-up rows where the largest lag/rolling window is undefined.
    # (These rows have NaN features and would otherwise leak NaNs into training.)
    drop_warmup_rows: bool = True

    # If set, keep only the first ``sample_items`` unique item_ids. Useful for a
    # fast end-to-end smoke test. ``None`` => use the full dataset.
    sample_items: int | None = None

    # Downcast numeric columns to save memory (the long frame is ~60M rows).
    downcast: bool = True

    @property
    def max_window(self) -> int:
        """Longest look-back used by any feature (for warm-up trimming)."""
        return max(self.lag_days + self.rolling_windows + [self.price_rolling_window])


# Default instance used by the CLI when no overrides are given.
DEFAULT_CONFIG = FeatureConfig()
