"""
M5 Evaluation Metrics
Implements WRMSSE (Weighted Root Mean Squared Scaled Error) — the official
M5 competition metric — plus RMSSE, MAE, MAPE, and SMAPE.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# RMSSE per series
# ─────────────────────────────────────────────────────────────

def rmsse(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    training_actuals: np.ndarray,
    h: int = 28,
) -> float:
    """
    Root Mean Squared Scaled Error for a single time series.

    RMSSE = sqrt( MSE(forecast) / MSE(naive) )
    where naive = sales_{t-1} (random walk).

    Args:
        actuals: shape (h,) ground truth
        forecasts: shape (h,) predictions
        training_actuals: training series for scale computation
        h: forecast horizon
    """
    # Scale: MSE of one-step-ahead naive forecast on training data
    naive_errors = np.diff(training_actuals)
    scale = np.mean(naive_errors ** 2)
    if scale < 1e-10:
        return 0.0

    mse = np.mean((actuals - forecasts) ** 2)
    return np.sqrt(mse / scale)


# ─────────────────────────────────────────────────────────────
# Series-level weights (dollar revenue weights)
# ─────────────────────────────────────────────────────────────

def compute_weights(
    sales_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    last_28_days_cols: List[str],
) -> pd.Series:
    """
    Compute per-series revenue weights for WRMSSE.
    Weight_i = sum(p_i * q_i over last 28 training days) / total revenue

    Returns:
        Series indexed by item-store id
    """
    # Get last-28-day sales
    sales_vals = sales_df[["id"] + last_28_days_cols].copy()
    sales_long = sales_vals.melt(id_vars="id", value_vars=last_28_days_cols,
                                  var_name="d", value_name="sales")

    # Join with calendar to get wm_yr_wk
    # Simplified: use average price
    item_store = sales_df[["id", "item_id", "store_id"]].copy()
    avg_price = prices_df.groupby(["item_id", "store_id"])["sell_price"].mean().reset_index()
    item_store = item_store.merge(avg_price, on=["item_id", "store_id"], how="left")
    item_store["sell_price"] = item_store["sell_price"].fillna(1.0)

    # Revenue = sum(sales) * avg_price
    total_sales = sales_long.groupby("id")["sales"].sum().reset_index()
    total_sales = total_sales.merge(item_store[["id", "sell_price"]], on="id", how="left")
    total_sales["revenue"] = total_sales["sales"] * total_sales["sell_price"]

    total_revenue = total_sales["revenue"].sum()
    total_sales["weight"] = total_sales["revenue"] / max(total_revenue, 1e-10)

    return total_sales.set_index("id")["weight"]


# ─────────────────────────────────────────────────────────────
# WRMSSE
# ─────────────────────────────────────────────────────────────

def wrmsse(
    actuals_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    training_df: pd.DataFrame,
    weights: pd.Series,
    h: int = 28,
) -> float:
    """
    Weighted RMSSE across all series.

    Args:
        actuals_df: [id, date, sales]
        forecasts_df: [id, date, forecast]
        training_df: [id, date, sales] — training history for scale
        weights: Series indexed by id
        h: forecast horizon

    Returns:
        WRMSSE scalar
    """
    ids = actuals_df["id"].unique()
    wrmsse_total = 0.0
    weight_total = 0.0

    for id_ in ids:
        act = actuals_df[actuals_df["id"] == id_].sort_values("date")["sales"].values
        fcast = forecasts_df[forecasts_df["id"] == id_].sort_values("date")["forecast"].values
        train = training_df[training_df["id"] == id_].sort_values("date")["sales"].values

        if len(act) == 0 or len(fcast) == 0 or len(train) < 2:
            continue

        r = rmsse(act[:h], fcast[:h], train)
        w = weights.get(id_, 0.0)
        wrmsse_total += w * r
        weight_total += w

    if weight_total > 0:
        return wrmsse_total / weight_total
    return 0.0


# ─────────────────────────────────────────────────────────────
# MAE / MAPE / SMAPE
# ─────────────────────────────────────────────────────────────

def mae(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    return float(np.mean(np.abs(actuals - forecasts)))


def mape(actuals: np.ndarray, forecasts: np.ndarray, eps: float = 1.0) -> float:
    """MAPE with epsilon to avoid division by zero on zero-demand items."""
    return float(np.mean(np.abs(actuals - forecasts) / np.maximum(np.abs(actuals), eps)) * 100)


def smape(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Symmetric MAPE."""
    denom = (np.abs(actuals) + np.abs(forecasts)) / 2 + 1e-10
    return float(np.mean(np.abs(actuals - forecasts) / denom) * 100)


def rmse(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actuals - forecasts) ** 2)))


# ─────────────────────────────────────────────────────────────
# Evaluation report
# ─────────────────────────────────────────────────────────────

def evaluate_forecasts(
    actuals_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    training_df: Optional[pd.DataFrame] = None,
    weights: Optional[pd.Series] = None,
    level_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute a comprehensive evaluation report by hierarchy level.

    Args:
        actuals_df: [id, date, sales] (+ optional level_col)
        forecasts_df: [id, date, forecast]
        training_df: history for RMSSE scaling
        weights: per-id weights for WRMSSE
        level_col: column to group by (e.g., 'store_id', 'cat_id')

    Returns:
        DataFrame with metrics per group
    """
    merged = actuals_df.merge(forecasts_df, on=["id", "date"], how="inner")
    if len(merged) == 0:
        logger.warning("No matching rows in actuals/forecasts!")
        return pd.DataFrame()

    # Overall metrics
    a = merged["sales"].values
    f = merged["forecast"].values
    results = []

    if level_col and level_col in merged.columns:
        for group, sub in merged.groupby(level_col):
            row = {
                "group": group,
                "level": level_col,
                "n_rows": len(sub),
                "MAE": mae(sub["sales"].values, sub["forecast"].values),
                "RMSE": rmse(sub["sales"].values, sub["forecast"].values),
                "MAPE": mape(sub["sales"].values, sub["forecast"].values),
                "SMAPE": smape(sub["sales"].values, sub["forecast"].values),
            }
            results.append(row)
    else:
        results.append({
            "group": "all",
            "level": "total",
            "n_rows": len(merged),
            "MAE": mae(a, f),
            "RMSE": rmse(a, f),
            "MAPE": mape(a, f),
            "SMAPE": smape(a, f),
        })

    return pd.DataFrame(results)


def compute_naive_baseline(
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    h: int = 28,
) -> pd.DataFrame:
    """
    Generate a naive (seasonal random walk, lag=7) baseline forecast.
    Used for WRMSSE improvement benchmarking.
    """
    last_week = train_df.sort_values("date").groupby("id").tail(7)
    naive_map = last_week.groupby("id")["sales"].mean()

    # Repeat naive for horizon
    val_dates = sorted(val_df["date"].unique())[:h]
    records = []
    for id_ in val_df["id"].unique():
        for d in val_dates:
            records.append({"id": id_, "date": d, "forecast": naive_map.get(id_, 0.0)})
    return pd.DataFrame(records)
