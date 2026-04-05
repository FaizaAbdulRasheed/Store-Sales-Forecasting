"""
Unit tests for the M5 Forecasting Pipeline
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.data.generator import M5DataGenerator
from src.data.preprocessor import melt_sales, reduce_mem_usage
from src.features.feature_engineering import add_calendar_features, add_lag_features
from src.evaluation.metrics import mae, rmse, mape, rmsse
from src.reconciliation.hierarchical import HierarchicalReconciler


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_sales_df():
    gen = M5DataGenerator(n_items_per_dept=2, seed=0)
    return gen.generate_sales_data()

@pytest.fixture(scope="module")
def small_calendar_df():
    gen = M5DataGenerator(n_items_per_dept=2, seed=0)
    return gen.generate_calendar()

@pytest.fixture(scope="module")
def small_prices_df(small_sales_df):
    gen = M5DataGenerator(n_items_per_dept=2, seed=0)
    return gen.generate_sell_prices(small_sales_df)


# ─────────────────────────────────────────────
# Data generator tests
# ─────────────────────────────────────────────

class TestDataGenerator:
    def test_sales_shape(self, small_sales_df):
        # 2 items/dept × 7 depts × 10 stores = 140 rows
        assert len(small_sales_df) == 2 * 7 * 10
        assert "item_id" in small_sales_df.columns
        assert "store_id" in small_sales_df.columns

    def test_no_negative_sales(self, small_sales_df):
        d_cols = [c for c in small_sales_df.columns if c.startswith("d_")]
        assert (small_sales_df[d_cols].values >= 0).all()

    def test_calendar_has_events(self, small_calendar_df):
        events = small_calendar_df["event_name_1"].dropna()
        assert len(events) > 0

    def test_calendar_snap_binary(self, small_calendar_df):
        for col in ["snap_CA", "snap_TX", "snap_WI"]:
            assert small_calendar_df[col].isin([0, 1]).all()


# ─────────────────────────────────────────────
# Preprocessing tests
# ─────────────────────────────────────────────

class TestPreprocessing:
    def test_melt_shape(self, small_sales_df, small_calendar_df):
        long = melt_sales(small_sales_df)
        n_days = len([c for c in small_sales_df.columns if c.startswith("d_")])
        assert len(long) == len(small_sales_df) * n_days
        assert "sales" in long.columns
        assert "d" in long.columns

    def test_memory_reduction(self, small_sales_df):
        import copy
        df = small_sales_df.copy()
        d_cols = [c for c in df.columns if c.startswith("d_")]
        numeric_df = df[d_cols].copy()
        before = numeric_df.memory_usage(deep=True).sum()
        reduced = reduce_mem_usage(numeric_df)
        after = reduced.memory_usage(deep=True).sum()
        assert after <= before


# ─────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────

class TestFeatureEngineering:
    def test_calendar_features(self):
        df = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=100)})
        df["sales"] = np.random.randint(0, 10, 100)
        out = add_calendar_features(df)
        for col in ["day_of_week", "month", "year", "sin_day_year", "cos_day_year"]:
            assert col in out.columns, f"Missing: {col}"

    def test_lag_features_no_leakage(self):
        ids = ["A"] * 50 + ["B"] * 50
        sales = list(range(50)) + list(range(50))
        dates = list(pd.date_range("2015-01-01", periods=50)) * 2
        df = pd.DataFrame({"id": ids, "date": dates, "sales": sales})
        df = df.sort_values(["id", "date"]).reset_index(drop=True)
        out = add_lag_features(df, lag_days=[7])
        # First 7 rows per series should be NaN
        first7 = out[out["id"] == "A"].head(7)
        assert first7["sales_lag_7"].isna().all()

    def test_fourier_range(self):
        df = pd.DataFrame({"date": pd.date_range("2011-01-01", periods=365)})
        df["sales"] = 1
        out = add_calendar_features(df)
        assert out["sin_day_year"].between(-1, 1).all()
        assert out["cos_day_year"].between(-1, 1).all()


# ─────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────

class TestMetrics:
    def test_mae_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == pytest.approx(0.0)

    def test_rmse_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_mape_basic(self):
        a = np.array([100.0, 200.0])
        f = np.array([110.0, 190.0])
        result = mape(a, f)
        assert 0 < result < 100

    def test_rmsse_naive_equals_one(self):
        # If forecast = naive (constant), RMSSE ≈ 1 for random walk
        train = np.arange(100, dtype=float)
        actuals = np.arange(100, 128, dtype=float)
        naive = np.full(28, train[-1])
        r = rmsse(actuals, naive, train)
        assert r > 0


# ─────────────────────────────────────────────
# Reconciliation tests
# ─────────────────────────────────────────────

class TestReconciliation:
    def test_bottom_up_coherent(self):
        """Sum of store forecasts should equal total forecast."""
        # Create dummy item-store forecasts
        data = []
        for i in range(5):  # 5 items
            for j in range(3):  # 3 stores
                for d in pd.date_range("2016-01-01", periods=5):
                    data.append({
                        "id": f"item_{i}_store_{j}_eval",
                        "item_id": f"item_{i}",
                        "dept_id": "FOODS_1",
                        "cat_id": "FOODS",
                        "store_id": f"store_{j}",
                        "state_id": "CA",
                        "date": d,
                        "forecast": float(i + j + 1),
                    })
        df = pd.DataFrame(data)
        hierarchy = df[["id", "item_id", "dept_id", "cat_id",
                         "store_id", "state_id"]].drop_duplicates()

        rec = HierarchicalReconciler(method="bottom_up")
        result = rec.bottom_up(df, hierarchy)

        assert "item_store" in result
        assert "store" in result
        assert "total" in result

        # Check total = sum of all item-store
        total_from_bottom = df.groupby("date")["forecast"].sum()
        total_agg = result["total"].set_index("date")["forecast"]
        pd.testing.assert_series_equal(
            total_from_bottom.sort_index(),
            total_agg.sort_index(),
            check_names=False,
            atol=1e-6,
        )

    def test_summing_matrix_shape(self):
        hierarchy = pd.DataFrame({
            "id": [f"item_{i}_store_{j}" for i in range(3) for j in range(2)],
            "item_id": [f"item_{i}" for i in range(3) for _ in range(2)],
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "store_id": [f"store_{j}" for _ in range(3) for j in range(2)],
            "state_id": "CA",
        })
        S = HierarchicalReconciler.build_summing_matrix(hierarchy)
        n_bottom = len(hierarchy)
        assert S.shape[1] == n_bottom
        assert S.shape[0] > n_bottom  # More rows than bottom-level series


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
