"""
Hierarchical Reconciliation Module
Implements Bottom-Up, Top-Down, OLS, and MinT reconciliation strategies
to ensure forecasts are coherent across the M5 hierarchy.

Reference: Hyndman et al. (2011) "Optimal combination forecasts for hierarchical time series"
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from scipy.linalg import pinv

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Hierarchy structure for M5
# ─────────────────────────────────────────────────────────────

M5_HIERARCHY = {
    "total": ["total"],
    "state": ["CA", "TX", "WI"],
    "store": ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"],
    "category": ["HOBBIES", "HOUSEHOLD", "FOODS"],
    "department": ["HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2", "FOODS_1", "FOODS_2", "FOODS_3"],
}


class HierarchicalReconciler:
    """
    Reconciles hierarchical forecasts using multiple strategies.

    Supports:
      - bottom_up: Aggregate bottom-level forecasts upward
      - top_down:  Disaggregate top-level forecast downward
      - ols:       Ordinary Least Squares regression-based reconciliation
      - mint:      Minimum Trace (shrinkage estimator)
    """

    def __init__(self, method: str = "bottom_up"):
        assert method in {"bottom_up", "top_down", "ols", "mint"}, \
            f"Unknown method: {method}. Choose from: bottom_up, top_down, ols, mint"
        self.method = method
        self._S: Optional[np.ndarray] = None  # Summing matrix

    # ─────────────────────────────────────────────────────────
    # Bottom-Up Reconciliation
    # ─────────────────────────────────────────────────────────

    def bottom_up(
        self,
        bottom_forecasts: pd.DataFrame,
        hierarchy_map: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate item-store forecasts bottom-up to all hierarchy levels.

        Args:
            bottom_forecasts: DataFrame with [id, item_id, dept_id, cat_id,
                              store_id, state_id, date, forecast]
            hierarchy_map: Item-store mapping table

        Returns:
            Dict mapping level_name → aggregated forecast DataFrame
        """
        logger.info("Running bottom-up reconciliation...")
        df = bottom_forecasts.copy()

        reconciled = {}
        reconciled["item_store"] = df[["id", "date", "forecast"]].copy()

        # State → Store → Category → Dept → Item aggregations
        for level, group_cols in [
            ("item", ["item_id", "date"]),
            ("dept_store", ["dept_id", "store_id", "date"]),
            ("dept", ["dept_id", "date"]),
            ("cat_store", ["cat_id", "store_id", "date"]),
            ("cat", ["cat_id", "date"]),
            ("store", ["store_id", "date"]),
            ("state", ["state_id", "date"]),
            ("total", ["date"]),
        ]:
            agg = df.groupby(group_cols)["forecast"].sum().reset_index()
            agg.columns = list(group_cols[:-1]) + ["date", "forecast"] if len(group_cols) > 2 \
                else group_cols + ["forecast"] if level == "total" \
                else group_cols + ["forecast"]
            if level == "total":
                agg = df.groupby("date")["forecast"].sum().reset_index()
                agg.columns = ["date", "forecast"]
            reconciled[level] = agg

        logger.info(f"Bottom-up complete. Levels: {list(reconciled.keys())}")
        return reconciled

    # ─────────────────────────────────────────────────────────
    # Top-Down Reconciliation
    # ─────────────────────────────────────────────────────────

    def top_down(
        self,
        top_forecast: pd.DataFrame,
        historical_proportions: pd.DataFrame,
        hierarchy_map: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Disaggregate top-level forecast using historical average proportions.

        Args:
            top_forecast: DataFrame with [date, forecast] (total level)
            historical_proportions: DataFrame with item-store proportions of total
            hierarchy_map: Item-store mapping

        Returns:
            Item-store level reconciled forecasts
        """
        logger.info("Running top-down reconciliation...")
        merged = top_forecast.merge(historical_proportions, on="date", how="left")
        merged["forecast_reconciled"] = merged["forecast"] * merged["proportion"]
        return merged[["id", "date", "forecast_reconciled"]].rename(
            columns={"forecast_reconciled": "forecast"}
        )

    @staticmethod
    def compute_proportions(
        historical_df: pd.DataFrame,
        recent_days: int = 365,
    ) -> pd.DataFrame:
        """
        Compute each item-store's average proportion of total sales.
        Used for top-down disaggregation.
        """
        max_date = historical_df["date"].max()
        cutoff = max_date - pd.Timedelta(days=recent_days)
        recent = historical_df[historical_df["date"] > cutoff]
        total_sales = recent.groupby("date")["sales"].sum().reset_index()
        total_sales.columns = ["date", "total_sales"]
        item_sales = recent.groupby(["id", "date"])["sales"].sum().reset_index()
        merged = item_sales.merge(total_sales, on="date")
        merged["proportion"] = merged["sales"] / merged["total_sales"].clip(lower=1e-6)
        avg_prop = merged.groupby("id")["proportion"].mean().reset_index()
        avg_prop.columns = ["id", "proportion"]
        # Normalise so they sum to 1
        avg_prop["proportion"] /= avg_prop["proportion"].sum()
        return avg_prop

    # ─────────────────────────────────────────────────────────
    # OLS Reconciliation
    # ─────────────────────────────────────────────────────────

    def ols_reconcile(
        self,
        base_forecasts: np.ndarray,
        S: np.ndarray,
    ) -> np.ndarray:
        """
        OLS-based reconciliation: ŷ_reconciled = S (S'S)^{-1} S' ŷ_base

        Args:
            base_forecasts: array of shape (n_series, n_horizons)
            S: summing matrix of shape (n_series, n_bottom)

        Returns:
            Reconciled forecasts of shape (n_series, n_horizons)
        """
        StS_inv = pinv(S.T @ S)
        P = StS_inv @ S.T  # (n_bottom, n_series)
        bottom_reconciled = P @ base_forecasts  # (n_bottom, n_horizons)
        return S @ bottom_reconciled  # (n_series, n_horizons)

    # ─────────────────────────────────────────────────────────
    # MinT (Minimum Trace) Reconciliation
    # ─────────────────────────────────────────────────────────

    def mint_reconcile(
        self,
        base_forecasts: np.ndarray,
        S: np.ndarray,
        residuals: np.ndarray,
        method: str = "shrink",
    ) -> np.ndarray:
        """
        MinT reconciliation using shrinkage estimator for covariance.
        ŷ_reconciled = S (S' Σ^{-1} S)^{-1} S' Σ^{-1} ŷ_base

        Args:
            base_forecasts: (n_series, n_horizons)
            S: summing matrix (n_series, n_bottom)
            residuals: training residuals (n_series, n_obs) for Σ estimation
            method: 'shrink' | 'sample' | 'ols' | 'wls'

        Returns:
            Reconciled forecasts (n_series, n_horizons)
        """
        Sigma = self._estimate_covariance(residuals, method=method)
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        StSiS = S.T @ Sigma_inv @ S
        try:
            StSiS_inv = np.linalg.inv(StSiS)
        except np.linalg.LinAlgError:
            StSiS_inv = np.linalg.pinv(StSiS)

        P = StSiS_inv @ S.T @ Sigma_inv
        bottom = P @ base_forecasts
        return S @ bottom

    @staticmethod
    def _estimate_covariance(
        residuals: np.ndarray,
        method: str = "shrink",
    ) -> np.ndarray:
        """
        Estimate forecast error covariance matrix.
        method: 'shrink' uses Ledoit-Wolf shrinkage for robustness.
        """
        n, T = residuals.shape
        if method == "shrink":
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(residuals.T)
            return lw.covariance_
        elif method == "sample":
            return np.cov(residuals)
        elif method == "ols":
            return np.eye(n)
        elif method == "wls":
            variances = np.var(residuals, axis=1)
            return np.diag(variances)
        else:
            return np.eye(n)

    # ─────────────────────────────────────────────────────────
    # Summing Matrix Builder
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_summing_matrix(hierarchy_map: pd.DataFrame) -> np.ndarray:
        """
        Build the summing matrix S for the M5 hierarchy.
        Rows = all series (total, state, store, cat, dept, item_store)
        Cols = bottom-level series (item_store)

        Returns:
            S matrix of shape (n_all_series, n_bottom)
        """
        bottom_ids = hierarchy_map["id"].tolist()
        n_bottom = len(bottom_ids)
        id_to_idx = {id_: i for i, id_ in enumerate(bottom_ids)}

        rows = []
        # Bottom level: identity
        for id_ in bottom_ids:
            row = np.zeros(n_bottom)
            row[id_to_idx[id_]] = 1
            rows.append(row)

        # Store level
        for store in hierarchy_map["store_id"].unique():
            row = np.zeros(n_bottom)
            for id_ in hierarchy_map[hierarchy_map["store_id"] == store]["id"]:
                row[id_to_idx[id_]] = 1
            rows.append(row)

        # State level
        for state in hierarchy_map["state_id"].unique():
            row = np.zeros(n_bottom)
            for id_ in hierarchy_map[hierarchy_map["state_id"] == state]["id"]:
                row[id_to_idx[id_]] = 1
            rows.append(row)

        # Category level
        for cat in hierarchy_map["cat_id"].unique():
            row = np.zeros(n_bottom)
            for id_ in hierarchy_map[hierarchy_map["cat_id"] == cat]["id"]:
                row[id_to_idx[id_]] = 1
            rows.append(row)

        # Total level
        total_row = np.ones(n_bottom)
        rows.append(total_row)

        S = np.vstack(rows)
        logger.info(f"Summing matrix S shape: {S.shape}")
        return S

    # ─────────────────────────────────────────────────────────
    # Coherence check
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def check_coherence(
        reconciled: Dict[str, pd.DataFrame],
        tolerance: float = 1.0,
    ) -> bool:
        """
        Verify that store-level forecasts sum to total within tolerance.
        """
        if "store" not in reconciled or "total" not in reconciled:
            return True

        store_total = reconciled["store"].groupby("date")["forecast"].sum().reset_index()
        total_df = reconciled["total"].rename(columns={"forecast": "total_forecast"})
        check = store_total.merge(total_df, on="date")
        diff = (check["forecast"] - check["total_forecast"]).abs()
        max_diff = diff.max()
        is_coherent = max_diff <= tolerance
        if not is_coherent:
            logger.warning(f"Coherence check failed: max diff = {max_diff:.4f}")
        else:
            logger.info(f"Coherence check passed: max diff = {max_diff:.4f}")
        return is_coherent
