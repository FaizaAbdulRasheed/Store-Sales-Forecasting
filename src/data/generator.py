"""
M5 Dataset Generator
Generates synthetic Walmart M5-like data for demonstration.
In production, replace with actual M5 Kaggle dataset.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    holidays = None

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]

CATEGORIES = {
    "HOBBIES": {"HOBBIES_1": 416, "HOBBIES_2": 290},
    "HOUSEHOLD": {"HOUSEHOLD_1": 712, "HOUSEHOLD_2": 235},
    "FOODS": {"FOODS_1": 249, "FOODS_2": 267, "FOODS_3": 880},
}

START_DATE = datetime(2011, 1, 29)
END_DATE = datetime(2016, 6, 19)

CULTURAL_EVENTS = [
    ("SuperBowl", "Sporting", "2011-02-06"), ("SuperBowl", "Sporting", "2012-02-05"),
    ("SuperBowl", "Sporting", "2013-02-03"), ("SuperBowl", "Sporting", "2014-02-02"),
    ("SuperBowl", "Sporting", "2015-02-01"), ("SuperBowl", "Sporting", "2016-02-07"),
    ("ValentinesDay", "Cultural", "2011-02-14"), ("ValentinesDay", "Cultural", "2012-02-14"),
    ("ValentinesDay", "Cultural", "2013-02-14"), ("ValentinesDay", "Cultural", "2014-02-14"),
    ("ValentinesDay", "Cultural", "2015-02-14"), ("ValentinesDay", "Cultural", "2016-02-14"),
    ("Easter", "Religious", "2011-04-24"), ("Easter", "Religious", "2012-04-08"),
    ("Easter", "Religious", "2013-03-31"), ("Easter", "Religious", "2014-04-20"),
    ("Easter", "Religious", "2015-04-05"), ("Easter", "Religious", "2016-03-27"),
    ("Halloween", "Cultural", "2011-10-31"), ("Halloween", "Cultural", "2012-10-31"),
    ("Halloween", "Cultural", "2013-10-31"), ("Halloween", "Cultural", "2014-10-31"),
    ("Halloween", "Cultural", "2015-10-31"),
    ("Thanksgiving", "National", "2011-11-24"), ("Thanksgiving", "National", "2012-11-22"),
    ("Thanksgiving", "National", "2013-11-28"), ("Thanksgiving", "National", "2014-11-27"),
    ("Thanksgiving", "National", "2015-11-26"),
    ("Christmas", "National", "2011-12-25"), ("Christmas", "National", "2012-12-25"),
    ("Christmas", "National", "2013-12-25"), ("Christmas", "National", "2014-12-25"),
    ("Christmas", "National", "2015-12-25"),
]


class M5DataGenerator:
    def __init__(self, n_items_per_dept: int = 5, seed: int = 42):
        self.seed = seed
        self.n_items_per_dept = n_items_per_dept
        np.random.seed(seed)

        self._build_item_catalog()
        self._build_date_range()

    def _build_item_catalog(self):
        items = []
        for cat, depts in CATEGORIES.items():
            for dept, full_count in depts.items():
                n = min(self.n_items_per_dept, full_count)
                for i in range(1, n + 1):
                    items.append({
                        "item_id": f"{dept}_{i:03d}",
                        "dept_id": dept,
                        "cat_id": cat,
                    })
        self.items_df = pd.DataFrame(items)

    def _build_date_range(self):
        self.dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
        self.n_days = len(self.dates)

    def _generate_base_demand(self, item_id: str, store_id: str) -> np.ndarray:
        rng = np.random.RandomState(hash(f"{item_id}_{store_id}") % (2**31))
        cat = item_id.split("_")[0]

        base_levels = {"HOBBIES": 1.5, "HOUSEHOLD": 4.0, "FOODS": 8.0}
        base = base_levels.get(cat, 3.0) * rng.uniform(0.5, 2.5)

        t = np.arange(self.n_days)

        trend = base * (1 + 0.0001 * t)

        day_of_week = np.array([d.dayofweek for d in self.dates])
        weekend_factor = 1 + 0.3 * (day_of_week >= 5).astype(float)

        annual = 1 + 0.2 * np.sin(2 * np.pi * t / 365.25)

        demand = trend * weekend_factor * annual

        noise = rng.negative_binomial(n=5, p=0.5, size=self.n_days) / 5
        demand *= noise

        zero_mask = rng.uniform(0, 1, self.n_days) < rng.uniform(0.05, 0.40)
        demand[zero_mask] = 0

        return np.maximum(demand, 0).astype(int)

    def _apply_holiday_effects(self, demand: np.ndarray, cat: str) -> np.ndarray:
        if not HOLIDAYS_AVAILABLE:
            return demand

        holiday_multipliers = {
            "FOODS": {"Christmas": 2.5, "default": 1.3},
            "HOBBIES": {"Christmas": 3.5, "default": 1.2},
            "HOUSEHOLD": {"Christmas": 2.0, "default": 1.2},
        }

        mults = holiday_multipliers.get(cat, {"default": 1.2})

        for name, _, date_str in CULTURAL_EVENTS:
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                idx = (pd.Timestamp(event_date) - pd.Timestamp(START_DATE)).days

                if 0 <= idx < self.n_days:
                    mult = mults.get(name, mults["default"])

                    for offset in range(-2, 3):
                        i = idx + offset
                        if 0 <= i < self.n_days:
                            demand[i] = int(demand[i] * mult)

            except Exception:
                continue

        return demand

    def generate_sales_data(self) -> pd.DataFrame:
        records = []
        d_cols = [f"d_{i}" for i in range(1, self.n_days + 1)]

        for _, item in self.items_df.iterrows():
            for store_id in STORES:
                demand = self._generate_base_demand(item["item_id"], store_id)
                demand = self._apply_holiday_effects(demand, item["cat_id"])

                row = {
                    "id": f"{item['item_id']}_{store_id}_evaluation",
                    "item_id": item["item_id"],
                    "dept_id": item["dept_id"],
                    "cat_id": item["cat_id"],
                    "store_id": store_id,
                }

                for j, d_col in enumerate(d_cols):
                    row[d_col] = demand[j]

                records.append(row)

        return pd.DataFrame(records)

    def generate_calendar(self) -> pd.DataFrame:
        rows = []

        for i, date in enumerate(self.dates):
            rows.append({
                "date": date,
                "weekday": date.strftime("%A"),
                "month": date.month,
                "year": date.year,
                "d": f"d_{i + 1}",
            })

        return pd.DataFrame(rows)

    def generate_sell_prices(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed)
        records = []

        for item_id in sales_df["item_id"].unique():
            for store_id in STORES:
                price = rng.uniform(1, 10)

                weeks = pd.date_range(start=START_DATE, end=END_DATE, freq="W")

                for wk in weeks:
                    if rng.random() < 0.05:
                        price *= rng.uniform(0.85, 1.15)

                    records.append({
                        "store_id": store_id,
                        "item_id": item_id,
                        "wm_yr_wk": int(wk.strftime("%G%V")),
                        "sell_price": round(price, 2),
                    })

        return pd.DataFrame(records)


def load_or_generate_data(n_items_per_dept: int = 8, seed: int = 42):
    gen = M5DataGenerator(n_items_per_dept=n_items_per_dept, seed=seed)

    sales_df = gen.generate_sales_data()
    calendar_df = gen.generate_calendar()
    prices_df = gen.generate_sell_prices(sales_df)

    return sales_df, calendar_df, prices_df