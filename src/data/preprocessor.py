"""
Data Preprocessing Pipeline
Converts M5 wide-format sales → long-format panel, merges calendar & prices,
applies memory optimisation, and builds the hierarchy mapping table.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Memory optimisation helpers
# ─────────────────────────────────────────────────────────────

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcast numeric columns to reduce RAM footprint."""
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != "category":
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            elif str(col_type)[:5] == "float":
                for dtype in [np.float16, np.float32, np.float64]:
                    if c_min >= np.finfo(dtype).min and c_max <= np.finfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        logger.info(f"Memory: {start_mem:.1f}MB → {end_mem:.1f}MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    return df


# ─────────────────────────────────────────────────────────────
# Core preprocessing
# ─────────────────────────────────────────────────────────────

def melt_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Wide → long (melts d_1 … d_N columns)."""
    all_id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    # Only keep id_cols that actually exist in the dataframe
    id_cols = [c for c in all_id_cols if c in sales_df.columns]
    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    long = sales_df.melt(id_vars=id_cols, value_vars=d_cols, var_name="d", value_name="sales")
    long["sales"] = long["sales"].astype(np.float32)
    return long


def merge_calendar(long_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar features to the long sales panel."""
    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    # Only select columns that actually exist
    cal_cols = [c for c in cal_cols if c in calendar_df.columns]
    cal = calendar_df[cal_cols].copy()
    merged = long_df.merge(cal, on="d", how="left")
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])
    return merged


def merge_prices(df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Attach sell price features."""
    merge_cols = [c for c in ["store_id", "item_id", "wm_yr_wk"] if c in df.columns and c in prices_df.columns]
    df = df.merge(prices_df, on=merge_cols, how="left")
    # Fill missing prices (items not yet available) with item-store mean
    df["sell_price"] = df.groupby(["item_id", "store_id"])["sell_price"].transform(
        lambda x: x.fillna(x.median())
    )
    df["sell_price"] = df["sell_price"].fillna(df["sell_price"].median())
    return df


def add_snap_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add unified SNAP flag based on state_id."""
    df["snap"] = 0
    if "snap_CA" in df.columns:
        df.loc[df["state_id"] == "CA", "snap"] = df.loc[df["state_id"] == "CA", "snap_CA"]
    if "snap_TX" in df.columns:
        df.loc[df["state_id"] == "TX", "snap"] = df.loc[df["state_id"] == "TX", "snap_TX"]
    if "snap_WI" in df.columns:
        df.loc[df["state_id"] == "WI", "snap"] = df.loc[df["state_id"] == "WI", "snap_WI"]
    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Label-encode categorical columns; return encoder map for inverse transform."""
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id",
                "weekday", "event_name_1", "event_type_1"]
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
            encoders[col] = dict(enumerate(df[col].cat.categories))
            df[col] = df[col].cat.codes.astype(np.int16)
    return df, encoders


def build_hierarchy_map(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Build a mapping table covering all M5 hierarchy levels."""
    all_id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    id_cols = [c for c in all_id_cols if c in sales_df.columns]
    return sales_df[id_cols].drop_duplicates().reset_index(drop=True)


def train_val_split(df: pd.DataFrame, val_days: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: last `val_days` form the validation set."""
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=val_days - 1)
    train = df[df["date"] < cutoff].copy()
    val = df[df["date"] >= cutoff].copy()
    return train, val


# ─────────────────────────────────────────────────────────────
# Full pipeline entry point
# ─────────────────────────────────────────────────────────────

def preprocess_pipeline(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    val_days: int = 28,
) -> Dict:
    """
    Full preprocessing pipeline.
    Returns a dict with keys: train, val, full, hierarchy, encoders
    """
    logger.info("Starting preprocessing pipeline...")

    # 1. Melt wide→long
    logger.info("Melting sales data...")
    long = melt_sales(sales_df)

    # 2. Merge calendar
    logger.info("Merging calendar...")
    long = merge_calendar(long, calendar_df)

    # 3. Merge prices
    logger.info("Merging prices...")
    long = merge_prices(long, prices_df)

    # 4. SNAP flag
    long = add_snap_flag(long)

    # 5. Build hierarchy map (before encoding)
    hierarchy = build_hierarchy_map(sales_df)

    # 6. Encode categoricals
    long, encoders = encode_categoricals(long)

    # 7. Memory optimisation
    long = reduce_mem_usage(long, verbose=True)

    # 8. Sort
    long = long.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # 9. Train/val split
    logger.info("Splitting train/val...")
    train, val = train_val_split(long, val_days=val_days)

    logger.info(f"Train: {len(train):,} rows | Val: {len(val):,} rows")

    return {
        "train": train,
        "val": val,
        "full": long,
        "hierarchy": hierarchy,
        "encoders": encoders,
    }
