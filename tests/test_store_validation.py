"""Tests validating store-wise forecasting integrity and metrics across all 10 M5 stores."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_FORECAST = ROOT / "data" / "model_ready" / "final_forecast_d1914_d1941.parquet"
FUTURE_FORECAST = ROOT / "data" / "model_ready" / "future_forecast_d1942_d1969.parquet"
EXPECTED_STORES = {"CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"}


class StoreValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FINAL_FORECAST.is_file():
            raise unittest.SkipTest(f"Historical forecast file {FINAL_FORECAST} not found.")
        cls.df = pd.read_parquet(FINAL_FORECAST)
        parsed = cls.df["id"].str.extract(
            r"^(?P<cat>[^_]+)_(?P<dept>[^_]+)_(?P<item>[^_]+)_(?P<state>CA|TX|WI)_(?P<store_num>[0-9]+)"
        )
        cls.df["store_id"] = parsed["state"] + "_" + parsed["store_num"]

    def test_all_ten_stores_present(self):
        found_stores = set(self.df["store_id"].unique())
        self.assertEqual(found_stores, EXPECTED_STORES)

    def test_row_counts_per_store(self):
        # 3,049 items * 28 days = 85,372 rows per store
        counts = self.df.groupby("store_id").size()
        for store in EXPECTED_STORES:
            self.assertEqual(counts[store], 85372, f"Store {store} expected 85,372 rows, got {counts[store]}")

    def test_predictions_non_negative_and_finite(self):
        self.assertFalse(self.df["predicted_sales"].isna().any(), "Found NaN in predicted_sales")
        self.assertFalse(self.df["actual_sales"].isna().any(), "Found NaN in actual_sales")
        self.assertTrue((self.df["predicted_sales"] >= 0).all(), "Found negative predicted_sales")
        self.assertTrue((self.df["actual_sales"] >= 0).all(), "Found negative actual_sales")

    def test_store_wise_error_bounds(self):
        for store, group in self.df.groupby("store_id"):
            mae = float(np.mean(np.abs(group["predicted_sales"] - group["actual_sales"])))
            rmse = float(np.sqrt(np.mean((group["predicted_sales"] - group["actual_sales"]) ** 2)))
            self.assertGreater(mae, 0.5, f"Store {store} MAE suspiciously low: {mae}")
            self.assertLess(mae, 2.0, f"Store {store} MAE exceeds threshold 2.0: {mae}")
            self.assertLess(rmse, 4.0, f"Store {store} RMSE exceeds threshold 4.0: {rmse}")

    def test_future_forecast_contains_all_stores(self):
        if not FUTURE_FORECAST.is_file():
            self.skipTest(f"Future forecast {FUTURE_FORECAST} not found.")
        future = pd.read_parquet(FUTURE_FORECAST)
        parsed = future["id"].str.extract(
            r"^(?P<cat>[^_]+)_(?P<dept>[^_]+)_(?P<item>[^_]+)_(?P<state>CA|TX|WI)_(?P<store_num>[0-9]+)"
        )
        future_stores = set((parsed["state"] + "_" + parsed["store_num"]).unique())
        self.assertEqual(future_stores, EXPECTED_STORES)
        self.assertEqual(len(future), 853720)


if __name__ == "__main__":
    unittest.main()
