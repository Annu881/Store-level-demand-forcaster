from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import data_quality
import direct_forecasting
import xgboost as xgb
from inventory import ReplenishmentInputs, simulate_replenishment

try:
    from fastapi.testclient import TestClient

    import api
except ImportError:
    TestClient = None
    api = None


class InventoryTests(unittest.TestCase):
    def test_rounding_and_protection_period(self) -> None:
        result = simulate_replenishment(
            [2.0] * 28,
            ReplenishmentInputs(
                on_hand=5,
                on_order=2,
                backorders=1,
                lead_time_days=2,
                review_period_days=1,
                pack_size=4,
                minimum_order_quantity=5,
                service_level=0.95,
                daily_residual_std=3,
            ),
        )
        self.assertAlmostEqual(result["inventory_position"], 6)
        self.assertAlmostEqual(result["lead_time_demand"], 4)
        self.assertAlmostEqual(result["protection_period_demand"], 6)
        self.assertAlmostEqual(result["lead_time_safety_stock"], 1.65 * np.sqrt(2) * 3)
        self.assertAlmostEqual(
            result["protection_period_safety_stock"],
            1.65 * np.sqrt(3) * 3,
        )
        self.assertEqual(result["suggested_order_quantity"], 12)

    def test_rejects_nonfinite_demand(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            simulate_replenishment(
                [float("nan")],
                ReplenishmentInputs(on_hand=0, on_order=0, backorders=0, lead_time_days=1),
            )

    def test_requires_full_protection_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "review period"):
            simulate_replenishment(
                [1, 1],
                ReplenishmentInputs(
                    on_hand=0,
                    on_order=0,
                    backorders=0,
                    lead_time_days=2,
                    review_period_days=1,
                ),
            )


class DataQualityTests(unittest.TestCase):
    def test_detects_duplicate_price_keys_across_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "prices.csv"
            pd.DataFrame(
                [
                    ("CA_1", "FOODS_1_001", 11101, 1.0),
                    ("CA_1", "FOODS_1_001", 11102, 1.0),
                    ("CA_1", "FOODS_1_001", 11101, 1.1),
                ],
                columns=["store_id", "item_id", "wm_yr_wk", "sell_price"],
            ).to_csv(path, index=False)
            result = data_quality._inspect_prices(
                path,
                ["store_id", "item_id", "wm_yr_wk", "sell_price"],
                chunk_size=1,
            )
        self.assertEqual(result[0], 3)
        self.assertEqual(result[1], 1)

    def test_validates_small_m5_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            sales_path = root / "sales.csv"
            calendar_path = root / "calendar.csv"
            prices_path = root / "prices.csv"
            pd.DataFrame(
                {
                    "id": ["FOODS_1_001_CA_1_evaluation"],
                    "item_id": ["FOODS_1_001"],
                    "dept_id": ["FOODS_1"],
                    "cat_id": ["FOODS"],
                    "store_id": ["CA_1"],
                    "state_id": ["CA"],
                    "d_1": [1],
                    "d_2": [0],
                }
            ).to_csv(sales_path, index=False)
            pd.DataFrame(
                {
                    "d": ["d_1", "d_2"],
                    "date": ["2011-01-29", "2011-01-30"],
                    "wm_yr_wk": [11101, 11101],
                    "wday": [1, 2],
                    "month": [1, 1],
                    "year": [2011, 2011],
                    "event_name_1": [None, None],
                    "event_name_2": [None, None],
                    "snap_CA": [0, 0],
                    "snap_TX": [0, 0],
                    "snap_WI": [0, 0],
                }
            ).to_csv(calendar_path, index=False)
            pd.DataFrame(
                {
                    "store_id": ["CA_1"],
                    "item_id": ["FOODS_1_001"],
                    "wm_yr_wk": [11101],
                    "sell_price": [1.0],
                }
            ).to_csv(prices_path, index=False)
            report = data_quality.validate_m5_inputs(
                sales_path,
                calendar_path,
                prices_path,
            )
        self.assertEqual(report["sales"]["series_count"], 1)
        self.assertEqual(report["prices"]["duplicate_price_keys"], 0)
        self.assertEqual(report["calendar"]["date_gap_count"], 0)


class DirectForecastFeatureTests(unittest.TestCase):
    def test_origin_features_use_only_observed_window(self) -> None:
        values = np.vstack(
            [
                np.arange(1, 61, dtype=np.float32),
                np.zeros(60, dtype=np.float32),
            ]
        )
        last_sale = direct_forecasting._last_sale_days(values)
        features = direct_forecasting._origin_dynamic_features(values, last_sale, 60)
        self.assertEqual(features["lag_1"][0], 60)
        self.assertEqual(features["lag_7"][0], 54)
        self.assertEqual(features["lag_56"][0], 5)
        self.assertEqual(features["days_since_last_sale"][0], 0)
        self.assertEqual(features["days_since_last_sale"][1], -1)
        self.assertEqual(features["no_previous_sale"][1], 1)

    def test_origin_schedule_includes_cutoff_without_target_leakage(self) -> None:
        origins = direct_forecasting._make_origin_list(1913)
        self.assertEqual(origins[0], 56)
        self.assertEqual(origins[-1], 1885)
        self.assertTrue(all(origin + 28 <= 1913 for origin in origins))
        with self.assertRaises(ValueError):
            direct_forecasting._make_origin_list(1913, step=0)

    def test_features_use_target_day_calendar_and_state_snap(self) -> None:
        day = pd.Series(
            {
                "wm_yr_wk": 11102,
                "wday": 3,
                "month": 2,
                "year": 2011,
                "day_of_week": 2,
                "day_of_month": 1,
                "week_of_year": 5,
                "month_num": 2,
                "year_num": 2011,
                "is_event": 0,
                "snap_CA": 1,
                "snap_TX": 0,
                "snap_WI": 1,
            }
        )
        calendar = pd.DataFrame([day], index=["d_57"])
        state_ids = ["CA", "TX", "WI"]
        dynamic = {
            name: np.zeros(3, dtype=np.float32)
            for name in direct_forecasting.DYNAMIC_FEATURES
        }
        dynamic["days_since_last_sale"] = np.asarray([0, -1, 4], dtype=np.float32)
        price = {name: np.ones(3, dtype=np.float32) for name in direct_forecasting.BASE_FEATURES if name in {
            "sell_price",
            "price_change",
            "price_change_pct",
            "is_price_change",
            "price_norm",
            "price_available",
        }}
        codes = {name: np.zeros(3, dtype=np.float32) for name in direct_forecasting.ENCODED_FEATURES[:-1]}
        features = direct_forecasting._features_for_horizon(
            56,
            1,
            dynamic,
            price,
            codes,
            calendar,
            state_ids,
        )
        self.assertEqual(features["wm_yr_wk"][0], 11102)
        self.assertEqual(features["snap"].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(features["horizon"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(features["days_since_last_sale"].tolist(), [1.0, -1.0, 5.0])

    def test_external_memory_iterator_trains_and_keeps_feature_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rng = np.random.default_rng(42)
            frame = pd.DataFrame(
                rng.random((8, len(direct_forecasting.DIRECT_FEATURES))),
                columns=direct_forecasting.DIRECT_FEATURES,
            )
            frame["sales"] = np.arange(8, dtype=np.float32)
            training_path = root / "training.parquet"
            frame.to_parquet(training_path, index=False)
            iterator = direct_forecasting.ParquetIterator(
                training_path,
                direct_forecasting.DIRECT_FEATURES,
                "sales",
                str(root / "cache"),
            )
            matrix = xgb.ExtMemQuantileDMatrix(iterator, max_bin=8)
            model = xgb.train(
                {
                    "objective": "count:poisson",
                    "tree_method": "hist",
                    "max_bin": 8,
                    "nthread": 1,
                },
                matrix,
                num_boost_round=1,
            )
            model.feature_names = direct_forecasting.DIRECT_FEATURES
            model_path = root / "model.json"
            model.save_model(model_path)
            loaded = xgb.Booster()
            loaded.load_model(model_path)
            prediction = loaded.predict(
                xgb.DMatrix(
                    frame[direct_forecasting.DIRECT_FEATURES].to_numpy(dtype=np.float32),
                    feature_names=direct_forecasting.DIRECT_FEATURES,
                )
            )
            del matrix, iterator
            gc.collect()
        self.assertEqual(len(prediction), 8)
        self.assertTrue(np.isfinite(prediction).all())


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are unavailable.")
class ApiTests(unittest.TestCase):
    def test_persists_and_retrieves_replenishment_simulation(self) -> None:
        assert api is not None
        assert TestClient is not None
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            direct_path = root / "direct.parquet"
            pd.DataFrame(
                {
                    "id": ["FOODS_1_001_CA_1_evaluation"] * 28,
                    "store_id": ["CA_1"] * 28,
                    "item_id": ["FOODS_1_001"] * 28,
                    "forecast_date_day": [f"d_{day}" for day in range(2, 30)],
                    "forecast_date": pd.date_range("2020-01-02", periods=28),
                    "horizon": list(range(1, 29)),
                    "point_forecast": [1.0] * 28,
                    "model_version": ["test"] * 28,
                    "data_version": ["test"] * 28,
                }
            ).to_parquet(direct_path, index=False)
            with (
                patch.dict(
                    api.FORECAST_PATHS,
                    {"direct": direct_path},
                ),
                patch.object(api, "DATABASE_FILE", root / "app.sqlite"),
                patch.object(api, "MODEL_READY", root),
            ):
                api._read_forecast.cache_clear()
                with TestClient(api.app) as client:
                    response = client.post(
                        "/replenishment",
                        json={
                            "store_id": "CA_1",
                            "item_id": "FOODS_1_001",
                            "forecast_type": "direct",
                            "lead_time_days": 7,
                            "on_hand": 0,
                        },
                    )
                    self.assertEqual(response.status_code, 201)
                    run_id = response.json()["run_id"]
                    self.assertEqual(
                        response.json()["result"]["suggested_order_quantity"],
                        7,
                    )
                    saved = client.get(f"/replenishment/{run_id}")
                    self.assertEqual(saved.status_code, 200)
                    self.assertEqual(saved.json()["item_id"], "FOODS_1_001")
                    invalid = client.post(
                        "/replenishment",
                        json={
                            "store_id": "CA_1",
                            "item_id": "FOODS_1_001",
                            "forecast_type": "direct",
                            "lead_time_days": 29,
                            "on_hand": 0,
                        },
                    )
                    self.assertEqual(invalid.status_code, 422)
                    invalid_service_level = client.post(
                        "/replenishment",
                        json={
                            "store_id": "CA_1",
                            "item_id": "FOODS_1_001",
                            "forecast_type": "direct",
                            "lead_time_days": 7,
                            "on_hand": 0,
                            "service_level": 0.94,
                        },
                    )
                    self.assertEqual(invalid_service_level.status_code, 422)
                api._read_forecast.cache_clear()


if __name__ == "__main__":
    unittest.main()
