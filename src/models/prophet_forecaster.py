"""
Prophet Forecaster
Trains per-item-store Prophet models for interpretable trend & seasonality
decomposition. Supports holiday injection and SNAP flags via regressors.
"""

import numpy as np
import pandas as pd
import logging
import os
import joblib
from typing import Dict, List, Optional, Tuple
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = logging.getLogger(__name__)


PROPHET_HOLIDAYS = pd.DataFrame([
    {"holiday": "SuperBowl", "ds": pd.Timestamp("2011-02-06"), "lower_window": -1, "upper_window": 1},
    {"holiday": "SuperBowl", "ds": pd.Timestamp("2012-02-05"), "lower_window": -1, "upper_window": 1},
    {"holiday": "SuperBowl", "ds": pd.Timestamp("2013-02-03"), "lower_window": -1, "upper_window": 1},
    {"holiday": "SuperBowl", "ds": pd.Timestamp("2014-02-02"), "lower_window": -1, "upper_window": 1},
    {"holiday": "SuperBowl", "ds": pd.Timestamp("2015-02-01"), "lower_window": -1, "upper_window": 1},
    {"holiday": "SuperBowl", "ds": pd.Timestamp("2016-02-07"), "lower_window": -1, "upper_window": 1},
    {"holiday": "Thanksgiving", "ds": pd.Timestamp("2011-11-24"), "lower_window": -2, "upper_window": 0},
    {"holiday": "Thanksgiving", "ds": pd.Timestamp("2012-11-22"), "lower_window": -2, "upper_window": 0},
    {"holiday": "Thanksgiving", "ds": pd.Timestamp("2013-11-28"), "lower_window": -2, "upper_window": 0},
    {"holiday": "Thanksgiving", "ds": pd.Timestamp("2014-11-27"), "lower_window": -2, "upper_window": 0},
    {"holiday": "Thanksgiving", "ds": pd.Timestamp("2015-11-26"), "lower_window": -2, "upper_window": 0},
    {"holiday": "Christmas", "ds": pd.Timestamp("2011-12-25"), "lower_window": -7, "upper_window": 0},
    {"holiday": "Christmas", "ds": pd.Timestamp("2012-12-25"), "lower_window": -7, "upper_window": 0},
    {"holiday": "Christmas", "ds": pd.Timestamp("2013-12-25"), "lower_window": -7, "upper_window": 0},
    {"holiday": "Christmas", "ds": pd.Timestamp("2014-12-25"), "lower_window": -7, "upper_window": 0},
    {"holiday": "Christmas", "ds": pd.Timestamp("2015-12-25"), "lower_window": -7, "upper_window": 0},
])


class ProphetForecaster:
    """
    Prophet wrapper for M5 hierarchical time series.
    Trains one model per aggregation level (e.g., per store, per dept).
    """

    DEFAULT_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "multiplicative",
}

    def __init__(self, params: Optional[Dict] = None, model_dir: str = "models/saved"):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model_dir = model_dir
        self.models: Dict[str, Prophet] = {}
        self.forecasts_: Dict[str, pd.DataFrame] = {}

    def _prepare_series(
        self, df: pd.DataFrame, group_col: str, group_val
    ) -> pd.DataFrame:
        """Extract and prepare a single time series for Prophet."""
        series = df[df[group_col] == group_val].copy()
        series = series.groupby("date")["sales"].sum().reset_index()
        series.columns = ["ds", "y"]
        series = series.sort_values("ds").reset_index(drop=True)
        # Log1p transform to handle zeros and stabilise variance
        series["y"] = np.log1p(series["y"].clip(lower=0))
        return series

    def fit_single(
        self,
        series: pd.DataFrame,
        series_id: str,
        add_snap: bool = False,
        snap_series: Optional[pd.Series] = None,
    ) -> Prophet:
        """Fit a single Prophet model."""
        try:
            m = Prophet(holidays=PROPHET_HOLIDAYS, **self.params)
        except Exception:
            # Fallback for older Prophet versions
            m = Prophet(holidays=PROPHET_HOLIDAYS)

        if add_snap and snap_series is not None:
            series = series.copy()
            series["snap"] = snap_series.values
            m.add_regressor("snap", standardize=False)

        m.fit(series)
        self.models[series_id] = m
        return m

    def fit_store_level(
        self,
        df: pd.DataFrame,
        stores: Optional[List[str]] = None,
        store_col: str = "store_id_label",
    ) -> "ProphetForecaster":
        """Fit one Prophet model per store. store_col can be label or code column."""
        if store_col not in df.columns:
            store_col = "store_id"
        if stores is None:
            stores = df[store_col].unique().tolist()

        logger.info(f"Fitting Prophet models for {len(stores)} stores (col={store_col!r})...")
        for store in stores:
            series = self._prepare_series(df, store_col, store)
            if len(series) < 60:
                logger.warning(f"Skipping {store}: insufficient data ({len(series)} rows)")
                continue
            self.fit_single(series, series_id=str(store))

        logger.info(f"Prophet store-level fitting complete. Models: {len(self.models)}")
        return self

    def fit_dept_level(
        self,
        df: pd.DataFrame,
        depts: Optional[List[str]] = None,
    ) -> "ProphetForecaster":
        """Fit one Prophet model per department."""
        if depts is None:
            depts = df["dept_id"].unique().tolist()
        logger.info(f"Fitting Prophet models for {len(depts)} departments...")
        for dept in depts:
            series = self._prepare_series(df, "dept_id", dept)
            if len(series) < 60:
                continue
            self.fit_single(series, series_id=f"dept_{dept}")
        logger.info("Prophet dept-level fitting complete.")
        return self

    def predict(
        self,
        series_id: str,
        horizon: int = 28,
    ) -> pd.DataFrame:
        """Generate Prophet forecast for a fitted series."""
        if series_id not in self.models:
            raise KeyError(f"No model for series_id={series_id!r}")
        m = self.models[series_id]
        future = m.make_future_dataframe(periods=horizon, freq="D")
        forecast = m.predict(future)
        # Inverse log1p transform
        forecast["yhat"] = np.expm1(forecast["yhat"]).clip(lower=0)
        forecast["yhat_lower"] = np.expm1(forecast["yhat_lower"]).clip(lower=0)
        forecast["yhat_upper"] = np.expm1(forecast["yhat_upper"]).clip(lower=0)
        self.forecasts_[series_id] = forecast
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper",
                          "trend", "weekly", "yearly"]].tail(horizon)

    def get_components(self, series_id: str) -> Optional[pd.DataFrame]:
        """Return decomposed forecast components."""
        return self.forecasts_.get(series_id)

    def forecast_all(self, horizon: int = 28) -> Dict[str, pd.DataFrame]:
        """Run predict() for all fitted series."""
        results = {}
        for sid in self.models.keys():
            results[sid] = self.predict(sid, horizon=horizon)
        return results

    def save(self, name: str = "prophet_forecaster"):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"{name}.pkl")
        joblib.dump({"models": self.models, "params": self.params}, path)
        logger.info(f"Prophet models saved to {path}")

    def load(self, name: str = "prophet_forecaster") -> "ProphetForecaster":
        path = os.path.join(self.model_dir, f"{name}.pkl")
        obj = joblib.load(path)
        self.models = obj["models"]
        self.params = obj["params"]
        logger.info(f"Prophet models loaded from {path}")
        return self
