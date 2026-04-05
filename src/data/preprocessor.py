"""
Data Preprocessing Pipeline - fully defensive against missing columns.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != "category":
            try:
                c_min, c_max = df[col].min(), df[col].max()
                if str(col_type)[:3] == "int":
                    for dtype in [np.int8, np.int16, np.int32, np.int64]:
                        if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                            df[col] = df[col].astype(dtype)
                            break
                elif str(col_type)[:5] == "float":
                    for dtype in [np.float32, np.float64]:
                        if c_min >= np.finfo(dtype).min and c_max <= np.finfo(dtype).max:
                            df[col] = df[col].astype(dtype)
                            break
            except Exception:
                pass
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        logger.info(f"Memory: {start_mem:.1f}MB -> {end_mem:.1f}MB")
    return df


def melt_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    desired_id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    id_cols = [c for c in desired_id_cols if c in sales_df.columns]
    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    long = sales_df.melt(id_vars=id_cols, value_vars=d_cols, var_name="d", value_name="sales")
    long["sales"] = long["sales"].astype(np.float32)
    return long


def merge_calendar(long_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    cal_cols = [c for c in desired if c in calendar_df.columns]
    cal = calendar_df[cal_cols].copy()
    merged = long_df.merge(cal, on="d", how="left")
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])
    return merged


def merge_prices(df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    desired_keys = ["store_id", "item_id", "wm_yr_wk"]
    merge_keys = [c for c in desired_keys if c in df.columns and c in prices_df.columns]
    if not merge_keys:
        df["sell_price"] = 1.0
        return df
    df = df.merge(prices_df, on=merge_keys, how="left")
    if "sell_price" in df.columns:
        group_cols = [c for c in ["item_id", "store_id"] if c in df.columns]
        if group_cols:
            df["sell_price"] = df.groupby(group_cols)["sell_price"].transform(
                lambda x: x.fillna(x.median())
            )
        df["sell_price"] = df["sell_price"].fillna(df["sell_price"].median()).fillna(1.0)
    else:
        df["sell_price"] = 1.0
    return df


def add_snap_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["snap"] = 0
    if "state_id" not in df.columns:
        return df
    for state, col in [("CA", "snap_CA"), ("TX", "snap_TX"), ("WI", "snap_WI")]:
        if col in df.columns:
            mask = df["state_id"] == state
            df.loc[mask, "snap"] = df.loc[mask, col]
    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
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
    desired = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    id_cols = [c for c in desired if c in sales_df.columns]
    return sales_df[id_cols].drop_duplicates().reset_index(drop=True)


def train_val_split(df: pd.DataFrame, val_days: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=val_days - 1)
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()


def preprocess_pipeline(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    val_days: int = 28,
) -> Dict:
    logger.info("Starting preprocessing pipeline...")
    logger.info(f"Sales shape: {sales_df.shape}")
    logger.info(f"Calendar cols: {list(calendar_df.columns)}")
    logger.info(f"Prices cols: {list(prices_df.columns)}")

    long = melt_sales(sales_df)
    logger.info(f"After melt: {long.shape}")

    long = merge_calendar(long, calendar_df)
    logger.info(f"After calendar: {long.shape}")

    long = merge_prices(long, prices_df)
    logger.info(f"After prices: {long.shape}")

    long = add_snap_flag(long)
    hierarchy = build_hierarchy_map(sales_df)
    long, encoders = encode_categoricals(long)
    long = reduce_mem_usage(long, verbose=True)

    sort_cols = [c for c in ["store_id", "item_id", "date"] if c in long.columns]
    long = long.sort_values(sort_cols).reset_index(drop=True)

    train, val = train_val_split(long, val_days=val_days)
    logger.info(f"Train: {len(train):,} | Val: {len(val):,}")

    return {"train": train, "val": val, "full": long, "hierarchy": hierarchy, "encoders": encoders}