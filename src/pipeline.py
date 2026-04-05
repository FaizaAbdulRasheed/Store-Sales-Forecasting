"""
Main Pipeline Orchestrator
End-to-end training and inference pipeline for M5 hierarchical forecasting.
Called by Streamlit app for on-demand training/inference with caching.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from src.data.generator import load_or_generate_data
from src.data.preprocessor import preprocess_pipeline
from src.features.feature_engineering import build_features, FEATURE_COLS
from src.models.lgbm_forecaster import LGBMForecaster
from src.models.prophet_forecaster import ProphetForecaster
from src.reconciliation.hierarchical import HierarchicalReconciler
from src.evaluation.metrics import (
    evaluate_forecasts, compute_naive_baseline,
    mae, rmse, mape, smape, wrmsse
)

logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """
    Full end-to-end pipeline:
    1. Data generation / loading
    2. Preprocessing
    3. Feature engineering
    4. LightGBM training + inference
    5. Prophet store-level training + inference
    6. Hierarchical reconciliation
    7. Evaluation
    """

    def __init__(
        self,
        n_items_per_dept: int = 6,
        forecast_horizon: int = 28,
        val_days: int = 28,
        lgbm_n_estimators: int = 200,
        seed: int = 42,
    ):
        self.n_items_per_dept = n_items_per_dept
        self.horizon = forecast_horizon
        self.val_days = val_days
        self.seed = seed

        self.lgbm = LGBMForecaster(params={"n_estimators": lgbm_n_estimators})
        self.prophet = ProphetForecaster()
        self.reconciler = HierarchicalReconciler(method="bottom_up")

        # State
        self.raw_data: Dict = {}
        self.processed: Dict = {}
        self.train_features: Optional[pd.DataFrame] = None
        self.val_features: Optional[pd.DataFrame] = None
        self.lgbm_preds: Optional[pd.DataFrame] = None
        self.prophet_preds: Dict = {}
        self.reconciled: Dict = {}
        self.metrics: Dict = {}
        self.is_trained = False

    # ─────────────────────────────────────────────────────────
    # Step 1: Load Data
    # ─────────────────────────────────────────────────────────

    def load_data(self, progress_callback=None) -> "ForecastingPipeline":
        t = time.time()
        logger.info("Step 1: Loading data...")
        if progress_callback:
            progress_callback(0.05, "Generating M5-like dataset...")

        sales_df, calendar_df, prices_df = load_or_generate_data(
            n_items_per_dept=self.n_items_per_dept, seed=self.seed
        )
        self.raw_data = {
            "sales": sales_df,
            "calendar": calendar_df,
            "prices": prices_df,
        }
        logger.info(f"Data loaded in {time.time()-t:.1f}s | Items: {len(sales_df)}")
        return self

    # ─────────────────────────────────────────────────────────
    # Step 2: Preprocess
    # ─────────────────────────────────────────────────────────

    def preprocess(self, progress_callback=None) -> "ForecastingPipeline":
        t = time.time()
        logger.info("Step 2: Preprocessing...")
        if progress_callback:
            progress_callback(0.15, "Preprocessing: melting, merging calendar & prices...")

        self.processed = preprocess_pipeline(
            self.raw_data["sales"],
            self.raw_data["calendar"],
            self.raw_data["prices"],
            val_days=self.val_days,
        )
        logger.info(f"Preprocessing done in {time.time()-t:.1f}s")
        return self

    # ─────────────────────────────────────────────────────────
    # Step 3: Feature Engineering
    # ─────────────────────────────────────────────────────────

    def engineer_features(self, progress_callback=None) -> "ForecastingPipeline":
        t = time.time()
        logger.info("Step 3: Feature engineering...")
        if progress_callback:
            progress_callback(0.30, "Engineering lag, rolling, price & calendar features...")

        train = self.processed["train"].sort_values(["id", "date"]).reset_index(drop=True)
        val = self.processed["val"].sort_values(["id", "date"]).reset_index(drop=True)

        # Build features on train
        self.train_features = build_features(train, training=True)
        # For val: build on combined then filter (to get correct lags)
        full = self.processed["full"].sort_values(["id", "date"]).reset_index(drop=True)
        full_features = build_features(full, training=False)
        max_train_date = train["date"].max()
        self.val_features = full_features[full_features["date"] > max_train_date].copy()

        logger.info(f"Feature engineering done in {time.time()-t:.1f}s "
                    f"| Train: {len(self.train_features):,} | Val: {len(self.val_features):,}")
        return self

    # ─────────────────────────────────────────────────────────
    # Step 4: Train LightGBM
    # ─────────────────────────────────────────────────────────

    def train_lgbm(self, progress_callback=None) -> "ForecastingPipeline":
        t = time.time()
        logger.info("Step 4: Training LightGBM...")
        if progress_callback:
            progress_callback(0.45, "Training LightGBM (Tweedie regression, global model)...")

        feature_cols = [c for c in FEATURE_COLS if c in self.train_features.columns]
        self.lgbm.fit(
            self.train_features,
            val_df=self.val_features,
            feature_cols=feature_cols,
            early_stopping_rounds=30,
        )
        # Predict on val
        self.lgbm_preds = self.val_features[["id", "date", "item_id", "dept_id",
                                              "cat_id", "store_id", "state_id"]].copy()
        self.lgbm_preds["forecast"] = self.lgbm.predict(self.val_features)

        logger.info(f"LightGBM training done in {time.time()-t:.1f}s")
        return self

    # ─────────────────────────────────────────────────────────
    # Step 5: Train Prophet (store level)
    # ─────────────────────────────────────────────────────────

    def train_prophet(self, progress_callback=None) -> "ForecastingPipeline":
        t = time.time()
        logger.info("Step 5: Training Prophet (store level)...")
        if progress_callback:
            progress_callback(0.60, "Training Prophet for each store...")

        # Use original (non-encoded) data for Prophet
        train_raw = self.processed["train"].copy()
        # Decode store_id from category codes
        # store_encoder maps {code_int: label_str}
        store_encoder = self.processed["encoders"].get("store_id", {})
        if store_encoder:
            # store_encoder is already {int_code: label_string}
            train_raw["store_id_label"] = train_raw["store_id"].map(store_encoder).fillna(train_raw["store_id"].astype(str))
        else:
            train_raw["store_id_label"] = train_raw["store_id"].astype(str)

        self.prophet.fit_store_level(train_raw, stores=train_raw["store_id_label"].unique().tolist(), store_col="store_id_label")
        # Generate forecasts
        self.prophet_preds = self.prophet.forecast_all(horizon=self.val_days)
        logger.info(f"Prophet training done in {time.time()-t:.1f}s "
                    f"| Models: {len(self.prophet.models)}")
        return self

    # ─────────────────────────────────────────────────────────
    # Step 6: Hierarchical Reconciliation
    # ─────────────────────────────────────────────────────────

    def reconcile(self, progress_callback=None) -> "ForecastingPipeline":
        t = time.time()
        logger.info("Step 6: Hierarchical reconciliation (bottom-up)...")
        if progress_callback:
            progress_callback(0.75, "Reconciling forecasts across hierarchy levels...")

        hierarchy_map = self.processed["hierarchy"]
        if self.lgbm_preds is not None:
            self.reconciled = self.reconciler.bottom_up(self.lgbm_preds, hierarchy_map)

        logger.info(f"Reconciliation done in {time.time()-t:.1f}s")
        return self

    # ─────────────────────────────────────────────────────────
    # Step 7: Evaluate
    # ─────────────────────────────────────────────────────────

    def evaluate(self, progress_callback=None) -> "ForecastingPipeline":
        t = time.time()
        logger.info("Step 7: Evaluating...")
        if progress_callback:
            progress_callback(0.88, "Computing WRMSSE, MAE, MAPE metrics...")

        val_actuals = self.processed["val"][["id", "date", "sales"]].copy()

        if self.lgbm_preds is not None:
            # Overall
            lgbm_eval = evaluate_forecasts(val_actuals, self.lgbm_preds)
            # By store
            val_meta = self.processed["val"][["id", "date", "store_id", "cat_id", "dept_id"]].drop_duplicates()
            val_with_store = val_actuals.merge(val_meta, on=["id", "date"], how="left")

            # lgbm_preds may or may not already have store_id — handle both cases
            preds_cols = ["id", "date", "forecast"]
            for col in ["store_id", "cat_id", "dept_id"]:
                if col in self.lgbm_preds.columns:
                    preds_cols.append(col)
            preds_with_store = self.lgbm_preds[preds_cols].copy()

            merged_eval = val_with_store.merge(
                preds_with_store, on=["id", "date"], how="inner"
            )

            # Ensure store_id exists — fall back to val_meta join if missing
            if "store_id" not in merged_eval.columns:
                merged_eval = merged_eval.merge(val_meta[["id", "date", "store_id", "cat_id", "dept_id"]],
                                                on=["id", "date"], how="left")
            # Naive baseline
            naive_preds = compute_naive_baseline(val_actuals, self.processed["train"])
            naive_eval = evaluate_forecasts(val_actuals, naive_preds)

            # Decode store/cat
            store_encoder = self.processed["encoders"].get("store_id", {})
            cat_encoder = self.processed["encoders"].get("cat_id", {})

            self.metrics = {
                "overall": lgbm_eval,
                "naive_overall": naive_eval,
                "lgbm_actuals": val_actuals,
                "lgbm_forecasts": self.lgbm_preds,
                "store_encoder": store_encoder,
                "cat_encoder": cat_encoder,
                "merged_eval": merged_eval,
            }

            # Compute summary metrics
            a = merged_eval["sales"].values
            f = merged_eval["forecast"].values
            self.metrics["summary"] = {
                "MAE": mae(a, f),
                "RMSE": rmse(a, f),
                "MAPE": mape(a, f),
                "SMAPE": smape(a, f),
            }
            # Naive
            naive_merged = val_actuals.merge(naive_preds, on=["id", "date"])
            n_a = naive_merged["sales"].values
            n_f = naive_merged["forecast"].values
            self.metrics["naive_summary"] = {
                "MAE": mae(n_a, n_f),
                "RMSE": rmse(n_a, n_f),
                "MAPE": mape(n_a, n_f),
                "SMAPE": smape(n_a, n_f),
            }

        logger.info(f"Evaluation done in {time.time()-t:.1f}s")
        return self

    # ─────────────────────────────────────────────────────────
    # Full run
    # ─────────────────────────────────────────────────────────

    def run(self, progress_callback=None) -> "ForecastingPipeline":
        """Execute the full pipeline end-to-end."""
        logger.info("=" * 60)
        logger.info("STARTING M5 FORECASTING PIPELINE")
        logger.info("=" * 60)
        t_total = time.time()

        (self
         .load_data(progress_callback)
         .preprocess(progress_callback)
         .engineer_features(progress_callback)
         .train_lgbm(progress_callback)
         .train_prophet(progress_callback)
         .reconcile(progress_callback)
         .evaluate(progress_callback))

        self.is_trained = True
        if progress_callback:
            progress_callback(1.0, "Pipeline complete!")
        logger.info(f"Full pipeline done in {time.time()-t_total:.1f}s")
        return self
