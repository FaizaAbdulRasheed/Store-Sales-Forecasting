"""
LightGBM Hierarchical Forecasting Model
Trains a single Tweedie-regression model over all items/stores,
leveraging rich features. Supports recursive multi-step forecasting.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import logging
import os
from typing import Dict, List, Optional, Tuple

from src.features.feature_engineering import FEATURE_COLS, build_features

logger = logging.getLogger(__name__)

TARGET = "sales"
FORECAST_HORIZON = 28


class LGBMForecaster:
    """
    LightGBM-based multi-step ahead forecaster.
    Single global model trained across all items and stores.
    """

    DEFAULT_PARAMS = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 20,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[Dict] = None, model_dir: str = "models/saved"):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model_dir = model_dir
        self.model: Optional[lgb.Booster] = None
        self.feature_importance_: Optional[pd.DataFrame] = None
        self._feature_cols: List[str] = []

    # ─────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        early_stopping_rounds: int = 50,
    ) -> "LGBMForecaster":
        """Train LightGBM on pre-featurised data."""
        feature_cols = feature_cols or [c for c in FEATURE_COLS if c in train_df.columns]
        self._feature_cols = feature_cols

        logger.info(f"Training LightGBM | features: {len(feature_cols)} | rows: {len(train_df):,}")

        X_train = train_df[feature_cols].astype(np.float32)
        y_train = train_df[TARGET].astype(np.float32)
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)

        callbacks = [lgb.log_evaluation(100)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if val_df is not None and len(val_df) > 0:
            X_val = val_df[feature_cols].astype(np.float32)
            y_val = val_df[TARGET].astype(np.float32)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        self.model = lgb.train(
            params={**self.params, "n_estimators": self.params.get("n_estimators", 1000)},
            train_set=dtrain,
            num_boost_round=self.params.get("n_estimators", 1000),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self._build_feature_importance()
        logger.info("LightGBM training complete.")
        return self

    def _build_feature_importance(self):
        if self.model is None:
            return
        importance = self.model.feature_importance(importance_type="gain")
        self.feature_importance_ = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Single-step prediction on a pre-featurised dataframe."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        available = [c for c in self._feature_cols if c in df.columns]
        X = df[available].astype(np.float32)
        return np.maximum(self.model.predict(X), 0)

    def forecast(
        self,
        history_df: pd.DataFrame,
        horizon: int = FORECAST_HORIZON,
    ) -> pd.DataFrame:
        """
        Recursive multi-step ahead forecast.
        Appends new rows for each future date, updating lags.
        Returns DataFrame with [id, date, forecast] columns.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        logger.info(f"Generating {horizon}-day forecast...")
        last_date = history_df["date"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

        ids = history_df["id"].unique()
        results = []

        for future_date in future_dates:
            # Build prediction frame for this date
            frame = history_df.drop_duplicates("id")[
                ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                 "sell_price", "snap", "has_event"]
            ].copy()
            frame["date"] = future_date
            frame["sales"] = 0  # placeholder

            # Re-engineer lags from the combined history+predictions so far
            combined = pd.concat([history_df, pd.DataFrame(results)] if results else [history_df],
                                  ignore_index=True)
            # Get lag values from combined
            combined_sorted = combined.sort_values(["id", "date"])
            for lag in [7, 14, 28, 35, 42]:
                lag_map = combined_sorted.groupby("id").apply(
                    lambda g: g.set_index("date")["sales"].shift(0)
                )
                # Simpler: look up directly
                pass

            # For production, use build_features on combined; here we use a simplified version
            frame = self._add_inference_features(frame, combined)
            preds = self.predict(frame)
            frame["forecast"] = preds
            results.append(frame[["id", "date", "forecast", "item_id", "dept_id",
                                   "cat_id", "store_id", "state_id"]])
            # Update history with predictions for next lag computation
            frame["sales"] = preds
            history_df = pd.concat([history_df, frame], ignore_index=True)

        return pd.concat(results, ignore_index=True)

    def _add_inference_features(self, frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """Add features for inference using historical data."""
        frame = frame.copy()
        hist_sorted = history.sort_values(["id", "date"])

        # Add lag features by looking back in history
        for lag in [7, 14, 28, 35, 42]:
            lag_date = frame["date"].iloc[0] - pd.Timedelta(days=lag)
            lag_vals = (
                hist_sorted[hist_sorted["date"] == lag_date]
                .set_index("id")["sales"]
            )
            frame[f"sales_lag_{lag}"] = frame["id"].map(lag_vals).fillna(0).astype(np.float32)

        # Rolling features
        for window in [7, 14, 28, 56]:
            cutoff = frame["date"].iloc[0] - pd.Timedelta(days=28)
            window_start = cutoff - pd.Timedelta(days=window)
            window_data = hist_sorted[
                (hist_sorted["date"] > window_start) & (hist_sorted["date"] <= cutoff)
            ]
            roll_mean = window_data.groupby("id")["sales"].mean()
            roll_std = window_data.groupby("id")["sales"].std()
            frame[f"rolling_mean_{window}"] = frame["id"].map(roll_mean).fillna(0).astype(np.float32)
            frame[f"rolling_std_{window}"] = frame["id"].map(roll_std).fillna(0).astype(np.float32)

        # Calendar features
        from src.features.feature_engineering import add_calendar_features, add_event_features
        frame = add_calendar_features(frame)
        frame = add_event_features(frame)

        # Store & dept mean (from last 7 days)
        recent = hist_sorted[hist_sorted["date"] >= hist_sorted["date"].max() - pd.Timedelta(days=7)]
        store_mean = recent.groupby("store_id")["sales"].mean()
        dept_mean = recent.groupby("dept_id")["sales"].mean()
        frame["store_daily_mean"] = frame["store_id"].map(store_mean).fillna(0).astype(np.float32)
        frame["dept_daily_mean"] = frame["dept_id"].map(dept_mean).fillna(0).astype(np.float32)

        # Price features (simplified)
        frame["price_roll_mean_28"] = frame["sell_price"].astype(np.float32)
        frame["price_change"] = np.float32(0)
        frame["price_relative"] = np.float32(1)
        frame["price_discount"] = np.float32(0)

        return frame

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def save(self, name: str = "lgbm_forecaster"):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"{name}.pkl")
        joblib.dump({"model": self.model, "feature_cols": self._feature_cols,
                     "params": self.params}, path)
        logger.info(f"Model saved to {path}")

    def load(self, name: str = "lgbm_forecaster") -> "LGBMForecaster":
        path = os.path.join(self.model_dir, f"{name}.pkl")
        obj = joblib.load(path)
        self.model = obj["model"]
        self._feature_cols = obj["feature_cols"]
        self.params = obj["params"]
        logger.info(f"Model loaded from {path}")
        return self
