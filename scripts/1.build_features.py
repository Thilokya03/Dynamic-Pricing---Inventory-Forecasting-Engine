"""Pipeline 1 entry point: build the feature table from the raw M5 data.

Reads the raw Kaggle M5 files from ``data/raw``, reshapes and merges them,
engineers lag / rolling / price-elasticity / calendar features, and writes a
single tidy parquet file to ``data/processed/features.parquet`` for the model
training pipeline to consume.

Usage
-----
    # full dataset
    python scripts/1.build_features.py

    # fast smoke test on the first 5 items
    python scripts/1.build_features.py --sample-items 5

    # custom output location
    python scripts/1.build_features.py --output data/processed/my_features.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make ``src`` importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features, config, data_loader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build M5 feature table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.FEATURES_PARQUET,
        help="Output parquet path (default: data/processed/features.parquet).",
    )
    parser.add_argument(
        "--sample-items",
        type=int,
        default=None,
        help="Keep only the first N item_ids (fast local test). Default: all.",
    )
    parser.add_argument(
        "--keep-warmup",
        action="store_true",
        help="Keep the warm-up rows that have NaN lag/rolling features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = config.FeatureConfig(
        sample_items=args.sample_items,
        drop_warmup_rows=not args.keep_warmup,
    )

    t0 = time.time()
    print("[1/3] Loading and merging raw M5 data ...")
    base = data_loader.build_base_frame(cfg)
    print(f"      base frame: {len(base):,} rows x {base.shape[1]} cols "
          f"({time.time() - t0:.1f}s)")

    print("[2/3] Engineering features ...")
    t1 = time.time()
    features = build_features.build_all_features(base, cfg)
    print(f"      feature frame: {len(features):,} rows x {features.shape[1]} cols "
          f"({time.time() - t1:.1f}s)")

    print("[3/3] Writing parquet ...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)

    size_mb = args.output.stat().st_size / 1e6
    print(f"      wrote {args.output} ({size_mb:.1f} MB)")
    print(f"Done in {time.time() - t0:.1f}s. "
          f"Features: {[c for c in features.columns if c not in config.ID_COLS]}")


if __name__ == "__main__":
    main()
