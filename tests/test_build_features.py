"""Unit tests for the feature builders.

These use a tiny hand-made frame so the expected lag / rolling / price values
can be checked by eye, and they guard against the most important bug class in
time-series features: leakage (a feature for day t using day t's own value).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features, config  # noqa: E402


def _toy_frame() -> pd.DataFrame:
    """Two series, 5 days each, with known sales and prices."""
    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    rows = []
    for series_id, state in [("A", "CA"), ("B", "TX")]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "id": series_id,
                    "state_id": state,
                    "date": date,
                    "sales": (i + 1) * (1 if series_id == "A" else 2),
                    "sell_price": 10.0 - i,  # price falls each day
                    "event_name_1": np.nan,
                    "snap_CA": 1,
                    "snap_TX": 0,
                    "snap_WI": 0,
                }
            )
    return pd.DataFrame(rows)


def test_lag_features_shift_per_series():
    df = build_features.add_lag_features(_toy_frame(), lags=[1])
    a = df[df["id"] == "A"].reset_index(drop=True)
    # day 0 has no prior day -> NaN; day 1's lag_1 == day 0's sales (=1).
    assert np.isnan(a.loc[0, "sales_lag_1"])
    assert a.loc[1, "sales_lag_1"] == 1
    assert a.loc[4, "sales_lag_1"] == 4
    # series B must not borrow series A's values.
    b = df[df["id"] == "B"].reset_index(drop=True)
    assert np.isnan(b.loc[0, "sales_lag_1"])


def test_rolling_mean_excludes_current_day():
    df = build_features.add_rolling_features(_toy_frame(), windows=[2])
    a = df[df["id"] == "A"].reset_index(drop=True)
    # roll_mean_2 on day 2 = mean(sales[0], sales[1]) = mean(1, 2) = 1.5
    # (current day excluded via shift(1)); proves no leakage.
    assert a.loc[2, "sales_roll_mean_2"] == 1.5


def test_price_features_detect_discount():
    df = build_features.add_price_features(_toy_frame(), price_window=3)
    a = df[df["id"] == "A"].reset_index(drop=True)
    # price drops by 1 each day, so price_change is negative.
    assert a.loc[1, "price_change"] == -1
    # discount vs rolling mean should be positive once price falls below avg.
    assert a["price_discount_pct"].iloc[-1] > 0


def test_calendar_snap_uses_correct_state():
    df = build_features.add_calendar_features(_toy_frame())
    # Series A is CA with snap_CA=1 -> active; series B is TX with snap_TX=0.
    assert (df[df["id"] == "A"]["snap_active"] == 1).all()
    assert (df[df["id"] == "B"]["snap_active"] == 0).all()


def test_build_all_runs_and_drops_warmup():
    cfg = config.FeatureConfig(
        lag_days=[1], rolling_windows=[2], price_rolling_window=2,
        drop_warmup_rows=True, downcast=False,
    )
    out = build_features.build_all_features(_toy_frame(), cfg)
    # max_window=2 -> first 2 rows of each 5-day series dropped => 3 each.
    assert len(out) == 6
    assert "sales_lag_1" in out.columns
    assert "price_elasticity" in out.columns
