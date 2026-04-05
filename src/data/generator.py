"""
M5 Dataset Generator
Generates synthetic Walmart M5-like data for demonstration.
Fully defensive — works with or without the 'holidays' package.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Optional holidays import
try:
    import holidays as holidays_lib
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    holidays_lib = None

# ─────────────────────────────────────────────────────────────
STORES = ["CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"]
STATES = {"CA":["CA_1","CA_2","CA_3","CA_4"],"TX":["TX_1","TX_2","TX_3"],"WI":["WI_1","WI_2","WI_3"]}

CATEGORIES = {
    "HOBBIES":    {"HOBBIES_1": 416, "HOBBIES_2": 290},
    "HOUSEHOLD":  {"HOUSEHOLD_1": 712, "HOUSEHOLD_2": 235},
    "FOODS":      {"FOODS_1": 249, "FOODS_2": 267, "FOODS_3": 880},
}

START_DATE = datetime(2011, 1, 29)
END_DATE   = datetime(2016, 6, 19)

CULTURAL_EVENTS = [
    ("SuperBowl",     "Sporting",  "2011-02-06"), ("SuperBowl",     "Sporting",  "2012-02-05"),
    ("SuperBowl",     "Sporting",  "2013-02-03"), ("SuperBowl",     "Sporting",  "2014-02-02"),
    ("SuperBowl",     "Sporting",  "2015-02-01"), ("SuperBowl",     "Sporting",  "2016-02-07"),
    ("ValentinesDay", "Cultural",  "2011-02-14"), ("ValentinesDay", "Cultural",  "2012-02-14"),
    ("ValentinesDay", "Cultural",  "2013-02-14"), ("ValentinesDay", "Cultural",  "2014-02-14"),
    ("ValentinesDay", "Cultural",  "2015-02-14"), ("ValentinesDay", "Cultural",  "2016-02-14"),
    ("Easter",        "Religious", "2011-04-24"), ("Easter",        "Religious", "2012-04-08"),
    ("Easter",        "Religious", "2013-03-31"), ("Easter",        "Religious", "2014-04-20"),
    ("Easter",        "Religious", "2015-04-05"), ("Easter",        "Religious", "2016-03-27"),
    ("MotherDay",     "Cultural",  "2011-05-08"), ("MotherDay",     "Cultural",  "2012-05-13"),
    ("MotherDay",     "Cultural",  "2013-05-12"), ("MotherDay",     "Cultural",  "2014-05-11"),
    ("MotherDay",     "Cultural",  "2015-05-10"), ("MotherDay",     "Cultural",  "2016-05-08"),
    ("Halloween",     "Cultural",  "2011-10-31"), ("Halloween",     "Cultural",  "2012-10-31"),
    ("Halloween",     "Cultural",  "2013-10-31"), ("Halloween",     "Cultural",  "2014-10-31"),
    ("Halloween",     "Cultural",  "2015-10-31"),
    ("Thanksgiving",  "National",  "2011-11-24"), ("Thanksgiving",  "National",  "2012-11-22"),
    ("Thanksgiving",  "National",  "2013-11-28"), ("Thanksgiving",  "National",  "2014-11-27"),
    ("Thanksgiving",  "National",  "2015-11-26"),
    ("Christmas",     "National",  "2011-12-25"), ("Christmas",     "National",  "2012-12-25"),
    ("Christmas",     "National",  "2013-12-25"), ("Christmas",     "National",  "2014-12-25"),
    ("Christmas",     "National",  "2015-12-25"),
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
                    items.append({"item_id": f"{dept}_{i:03d}", "dept_id": dept, "cat_id": cat})
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
        weekend_factor = 1 + 0.3 * (day_of_week >= 5).astype(float) * rng.uniform(0.5, 1.5)
        annual = 1 + 0.2 * np.sin(2 * np.pi * t / 365.25 + rng.uniform(0, 2 * np.pi))
        demand = trend * weekend_factor * annual
        noise = rng.negative_binomial(n=5, p=0.5, size=self.n_days) / 5
        demand = demand * noise
        zero_prob = rng.uniform(0.05, 0.40)
        zero_mask = rng.uniform(0, 1, self.n_days) < zero_prob
        demand[zero_mask] = 0
        return np.maximum(demand, 0).astype(int)

    def _apply_holiday_effects(self, demand: np.ndarray, cat: str) -> np.ndarray:
        holiday_multipliers = {
            "FOODS":     {"Christmas": 2.5, "Thanksgiving": 3.0, "SuperBowl": 1.8, "Easter": 1.5, "Halloween": 1.4, "default": 1.3},
            "HOBBIES":   {"Christmas": 3.5, "Easter": 1.3, "Halloween": 2.0, "ValentinesDay": 1.6, "default": 1.2},
            "HOUSEHOLD": {"Christmas": 2.0, "MotherDay": 1.8, "default": 1.2},
        }
        mults = holiday_multipliers.get(cat, {"default": 1.2})
        for name, event_type, date_str in CULTURAL_EVENTS:
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                idx = (pd.Timestamp(event_date) - pd.Timestamp(START_DATE)).days
                if 0 <= idx < self.n_days:
                    mult = mults.get(name, mults.get("default", 1.2))
                    for offset in range(-2, 3):
                        i = idx + offset
                        if 0 <= i < self.n_days:
                            decay = 1 - 0.2 * abs(offset)
                            demand[i] = int(demand[i] * mult * decay)
            except Exception:
                continue
        return demand

    def generate_sales_data(self) -> pd.DataFrame:
        logger.info("Generating sales data...")
        records = []
        d_cols = [f"d_{i}" for i in range(1, self.n_days + 1)]
        for _, item in self.items_df.iterrows():
            for store_id in STORES:
                state_id = store_id.split("_")[0]
                demand = self._generate_base_demand(item["item_id"], store_id)
                demand = self._apply_holiday_effects(demand, item["cat_id"])
                row = {
                    "id": f"{item['item_id']}_{store_id}_evaluation",
                    "item_id": item["item_id"],
                    "dept_id": item["dept_id"],
                    "cat_id": item["cat_id"],
                    "store_id": store_id,
                    "state_id": state_id,
                }
                for j, d_col in enumerate(d_cols):
                    row[d_col] = int(demand[j])
                records.append(row)
        df = pd.DataFrame(records)
        logger.info(f"Sales data shape: {df.shape}")
        return df

    def generate_calendar(self) -> pd.DataFrame:
        """Generate M5 calendar — defensive, works with or without holidays package."""
        # Build event lookup from our hardcoded list
        event_dict = {}
        for name, event_type, date_str in CULTURAL_EVENTS:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                event_dict[dt] = (name, event_type)
            except Exception:
                continue

        # Build US holidays lookup if package available
        us_holidays = {}
        if HOLIDAYS_AVAILABLE and holidays_lib is not None:
            try:
                us_holidays = holidays_lib.US(years=range(2011, 2017))
            except Exception:
                us_holidays = {}

        rows = []
        for i, date in enumerate(self.dates):
            d = date.date()

            # Event from our list
            event_name_1 = event_dict.get(d, (None, None))[0]
            event_type_1 = event_dict.get(d, (None, None))[1]

            # Fall back to holidays package
            if event_name_1 is None and us_holidays:
                holiday_name = us_holidays.get(d)
                if holiday_name:
                    event_name_1 = holiday_name
                    event_type_1 = "National"

            # Safe week number
            try:
                wm_yr_wk = int(date.strftime("%G%V"))
            except Exception:
                wm_yr_wk = int(date.strftime("%Y%W"))

            rows.append({
                "date":         date,
                "wm_yr_wk":    wm_yr_wk,
                "weekday":      date.strftime("%A"),
                "wday":         date.dayofweek + 1,
                "month":        date.month,
                "year":         date.year,
                "d":            f"d_{i + 1}",
                "event_name_1": event_name_1,
                "event_type_1": event_type_1,
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA":      int(date.day in [1, 2, 3, 4, 5, 6]),
                "snap_TX":      int(date.day in [1, 2, 3, 4, 5, 6, 7]),
                "snap_WI":      int(date.day in [1, 2, 3, 4, 5, 6, 7, 8]),
            })

        cal = pd.DataFrame(rows)
        logger.info(f"Calendar shape: {cal.shape}, cols: {list(cal.columns)}")
        return cal

    def generate_sell_prices(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed)
        records = []
        for item_id in sales_df["item_id"].unique():
            for store_id in STORES:
                cat = item_id.split("_")[0]
                base_prices = {"HOBBIES": 8.0, "HOUSEHOLD": 5.5, "FOODS": 2.5}
                base_price = base_prices.get(cat, 4.0) * rng.uniform(0.5, 3.0)
                wks = pd.date_range(start=START_DATE, end=END_DATE, freq="W-SAT")
                price = base_price
                for wk in wks:
                    if rng.random() < 0.05:
                        price = max(0.50, price * rng.uniform(0.85, 1.15))
                    try:
                        wm_yr_wk = int(wk.strftime("%G%V"))
                    except Exception:
                        wm_yr_wk = int(wk.strftime("%Y%W"))
                    records.append({
                        "store_id":   store_id,
                        "item_id":    item_id,
                        "wm_yr_wk":  wm_yr_wk,
                        "sell_price": round(price, 2),
                    })
        return pd.DataFrame(records)


def load_or_generate_data(n_items_per_dept: int = 8, seed: int = 42):
    gen = M5DataGenerator(n_items_per_dept=n_items_per_dept, seed=seed)
    sales_df    = gen.generate_sales_data()
    calendar_df = gen.generate_calendar()
    prices_df   = gen.generate_sell_prices(sales_df)
    return sales_df, calendar_df, prices_df