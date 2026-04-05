"""
Feature Engineering Pipeline
Calendar-aware features, lag/rolling features, price features,
and event-based demand indicators for LightGBM.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Calendar features
# ─────────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rich calendar features."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek.astype(np.int8)
    df["day_of_month"] = df["date"].dt.day.astype(np.int8)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(np.int16)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int8)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["quarter"] = df["date"].dt.quarter.astype(np.int8)
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(np.int8)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(np.int8)
    df["is_quarter_start"] = df["date"].dt.is_quarter_start.astype(np.int8)
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(np.int8)

    # Fourier features for yearly & weekly seasonality
    df["sin_day_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
    df["cos_day_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
    df["sin_day_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)
    df["cos_day_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)

    return df


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode M5 events as binary flags + days-since/until features.
    event_name_1 must already be label-encoded or None.
    """
    df = df.copy()
    # Binary: any event today
    df["has_event"] = (df["event_name_1"].notna() & (df["event_name_1"] != -1)).astype(np.int8)
    # Proxy: event tomorrow / yesterday (shift if sorted by item-store-date)
    return df


# ─────────────────────────────────────────────────────────────
# Lag & rolling features
# ─────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    lag_days: List[int] = [7, 14, 28, 35, 42],
    group_cols: List[str] = ["id"],
) -> pd.DataFrame:
    """
    Compute lag features on the sales column.
    IMPORTANT: data must be sorted by [id, date] before calling.
    """
    df = df.copy()
    for lag in lag_days:
        col_name = f"sales_lag_{lag}"
        df[col_name] = df.groupby(group_cols)["sales"].shift(lag).astype(np.float32)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [7, 14, 28, 56],
    lag: int = 28,
    group_cols: List[str] = ["id"],
) -> pd.DataFrame:
    """
    Rolling mean & std features (computed on lagged series to prevent leakage).
    """
    df = df.copy()
    shifted = df.groupby(group_cols)["sales"].shift(lag)
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            shifted.groupby(df[group_cols[0]]).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            ).astype(np.float32)
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby(df[group_cols[0]]).transform(
                lambda x: x.rolling(window, min_periods=1).std()
            ).astype(np.float32)
        )
    return df


def add_rolling_features_fast(
    df: pd.DataFrame,
    windows: List[int] = [7, 14, 28, 56],
    group_cols: List[str] = ["id"],
) -> pd.DataFrame:
    """
    Faster rolling features using shift(28) then rolling — avoids group lambda.
    """
    df = df.copy()
    df["_shifted_sales"] = df.groupby(group_cols)["sales"].shift(28)
    for w in windows:
        df[f"rolling_mean_{w}"] = (
            df.groupby(group_cols)["_shifted_sales"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
            .astype(np.float32)
        )
        df[f"rolling_std_{w}"] = (
            df.groupby(group_cols)["_shifted_sales"]
            .transform(lambda x: x.rolling(w, min_periods=1).std())
            .astype(np.float32)
        )
    df.drop(columns=["_shifted_sales"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# Price features
# ─────────────────────────────────────────────────────────────

def add_price_features(
    df: pd.DataFrame,
    group_cols: List[str] = ["item_id", "store_id"],
) -> pd.DataFrame:
    """Price-based features capturing elasticity signals."""
    df = df.copy()
    # Rolling average price (28-day)
    df["price_roll_mean_28"] = (
        df.groupby(group_cols)["sell_price"]
        .transform(lambda x: x.rolling(28, min_periods=1).mean())
        .astype(np.float32)
    )
    # Price change flag
    df["price_change"] = (
        df.groupby(group_cols)["sell_price"]
        .transform(lambda x: x.diff().fillna(0))
        .astype(np.float32)
    )
    # Price normalised by item mean (relative price level)
    item_mean_price = df.groupby(group_cols)["sell_price"].transform("mean")
    df["price_relative"] = (df["sell_price"] / item_mean_price).astype(np.float32)
    # Max price (captures promotion depth)
    df["price_max"] = (
        df.groupby(group_cols)["sell_price"]
        .transform("max")
        .astype(np.float32)
    )
    df["price_discount"] = (
        ((df["price_max"] - df["sell_price"]) / df["price_max"].clip(lower=0.01))
        .astype(np.float32)
    )
    return df


# ─────────────────────────────────────────────────────────────
# Store-level aggregations
# ─────────────────────────────────────────────────────────────

def add_store_aggregate_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add store-level rolling demand context."""
    df = df.copy()
    store_mean = df.groupby(["store_id", "date"])["sales"].transform("mean")
    df["store_daily_mean"] = store_mean.astype(np.float32)
    dept_mean = df.groupby(["dept_id", "date"])["sales"].transform("mean")
    df["dept_daily_mean"] = dept_mean.astype(np.float32)
    return df


# ─────────────────────────────────────────────────────────────
# Full feature pipeline
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Calendar
    "day_of_week", "day_of_month", "day_of_year", "week_of_year",
    "month", "quarter", "year", "is_weekend",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "sin_day_year", "cos_day_year", "sin_day_week", "cos_day_week",
    # Events
    "has_event", "snap",
    # IDs (encoded)
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    # Lags
    "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_lag_35", "sales_lag_42",
    # Rolling
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_mean_56",
    "rolling_std_7", "rolling_std_14", "rolling_std_28", "rolling_std_56",
    # Price
    "sell_price", "price_roll_mean_28", "price_change",
    "price_relative", "price_discount",
    # Store aggregations
    "store_daily_mean", "dept_daily_mean",
]


def build_features(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Args:
        df: Long-format sales panel (sorted by [id, date]).
        training: If True, may drop rows with NaN lags.
    Returns:
        Feature-enriched DataFrame.
    """
    logger.info("Building calendar features...")
    df = add_calendar_features(df)
    logger.info("Building event features...")
    df = add_event_features(df)
    logger.info("Building lag features...")
    df = add_lag_features(df, lag_days=[7, 14, 28, 35, 42])
    logger.info("Building rolling features...")
    df = add_rolling_features_fast(df, windows=[7, 14, 28, 56])
    logger.info("Building price features...")
    df = add_price_features(df)
    logger.info("Building store aggregate features...")
    df = add_store_aggregate_features(df)

    # Drop rows where lags can't be computed (training only)
    if training:
        df = df.dropna(subset=["sales_lag_28", "rolling_mean_28"]).reset_index(drop=True)

    logger.info(f"Feature engineering done. Shape: {df.shape}")
    return df
