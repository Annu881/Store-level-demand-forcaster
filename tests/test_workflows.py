from __future__ import annotations

import gc
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import data_quality
import direct_forecasting
import forecasting_service
import training_jobs
import training_job_runner
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

    def test_replenishment_with_uploaded_series_and_xgboost_forecasts(self) -> None:
        """Verify replenishment simulator calculates correct order quantities

        for uploaded series IDs (FOODS_1_001_CA_1, FOODS_1_002_CA_1, HOBBIES_1_001_TX_1)
        using their saved XGBoost forecast demand and inventory inputs.
        """
        uploaded_series = [
            "FOODS_1_001_CA_1",
            "FOODS_1_002_CA_1",
            "HOBBIES_1_001_TX_1",
        ]
        # Load saved user forecast result
        saved = forecasting_service.load_latest_forecast()
        self.assertIsNotNone(saved, "Saved forecast must exist for simulation tests.")
        forecast_df, actual_df, _ = saved

        for sid in uploaded_series:
            sub = forecast_df.loc[forecast_df["id"] == sid].sort_values("date")
            self.assertFalse(sub.empty, f"Series {sid} must have forecasts.")
            daily_forecasts = sub["predicted_sales"].astype(float).tolist()
            self.assertGreaterEqual(len(daily_forecasts), 28)

            # Test replenishment simulation with realistic inputs
            inputs = ReplenishmentInputs(
                on_hand=30.0,
                on_order=10.0,
                backorders=0.0,
                lead_time_days=7,
                review_period_days=0,
                pack_size=4,
                minimum_order_quantity=10,
                service_level=0.95,
                daily_residual_std=2.0,
            )
            result = simulate_replenishment(daily_forecasts, inputs)

            # Check math integrity
            expected_pos = 40.0  # on_hand (30) + on_order (10) - backorders (0)
            self.assertAlmostEqual(result["inventory_position"], expected_pos)

            expected_7d_demand = sum(daily_forecasts[:7])
            self.assertAlmostEqual(result["lead_time_demand"], expected_7d_demand)
            self.assertAlmostEqual(result["protection_period_demand"], expected_7d_demand)

            expected_safety_stock = 1.65 * np.sqrt(7) * 2.0
            self.assertAlmostEqual(result["safety_stock"], expected_safety_stock, places=4)
            self.assertAlmostEqual(result["reorder_point"], expected_7d_demand + expected_safety_stock, places=4)

            # Order quantity must respect pack size multiples and MOQ
            order_qty = result["suggested_order_quantity"]
            self.assertEqual(order_qty % 4, 0, "Order qty must be multiple of pack size (4).")
            if order_qty > 0:
                self.assertGreaterEqual(order_qty, 10, "Order qty must satisfy MOQ (10).")


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

    def test_distribution_drift_detection(self) -> None:
        normal_sales = pd.DataFrame({"sales": [1, 2, 0, 0, 1, 3, 0, 2, 0, 1]})
        normal_result = data_quality.check_sales_distribution_drift(normal_sales)
        self.assertEqual(normal_result["status"], "normal")
        self.assertFalse(normal_result["drift_detected"])

        drifting_sales = pd.DataFrame({"sales": [50, 60, 45, 80, 55, 70]})
        drift_result = data_quality.check_sales_distribution_drift(drifting_sales)
        self.assertEqual(drift_result["status"], "warning")
        self.assertTrue(drift_result["drift_detected"])
        self.assertTrue(len(drift_result["warnings"]) > 0)


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


class ForecastingServiceTests(unittest.TestCase):
    def test_validates_uploaded_sales_and_rejects_duplicate_days(self) -> None:
        rows = forecasting_service.validate_sales_rows(
            pd.DataFrame({
                "id": ["series-a"],
                "date": ["2026-01-01"],
                "sales": ["3"],
            })
        )
        self.assertEqual(rows.loc[0, "sales"], 3)
        self.assertTrue(pd.isna(rows.loc[0, "sell_price"]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            forecasting_service.validate_sales_rows(
                pd.DataFrame({
                    "id": ["series-a", "series-a"],
                    "date": ["2026-01-01", "2026-01-01"],
                    "sales": [1, 2],
                })
            )
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            forecasting_service.validate_sales_rows(
                pd.DataFrame({
                    "id": ["series-a"],
                    "date": ["2026-01-01"],
                    "sales": [1],
                    "sell_price": ["invalid"],
                })
            )

    def test_calendar_fallback_derives_fields_without_reference_calendar(self) -> None:
        features = forecasting_service.calendar_features_for_date(
            pd.Timestamp("2026-03-04"),
            pd.DataFrame(),
        )
        self.assertEqual(features["month"], 3.0)
        self.assertEqual(features["day_of_month"], 4.0)
        self.assertEqual(features["is_event"], 0.0)
        self.assertEqual(features["snap_CA"], 0.0)

    def test_merges_updates_by_series_and_date(self) -> None:
        previous = pd.DataFrame({
            "id": ["series-a", "series-a"],
            "date": ["2026-01-01", "2026-01-02"],
            "sales": [1, 2],
            "sell_price": [4.0, 4.0],
        })
        update = pd.DataFrame({
            "id": ["series-a"],
            "date": ["2026-01-02"],
            "sales": [7],
        })
        merged = forecasting_service.merge_sales_history(previous, update)
        self.assertEqual(len(merged), 2)
        self.assertEqual(
            merged.loc[merged["date"].eq(pd.Timestamp("2026-01-02")), "sales"].iloc[0],
            7,
        )
        self.assertEqual(merged["sell_price"].tolist(), [4.0, 4.0])

    def test_joblib_bundle_and_future_forecast_uses_xgboost_for_all_series(self) -> None:
        features = forecasting_service._feature_names()
        train_data = np.random.default_rng(42).random(
            (16, len(features))
        ).astype(np.float32)
        train_labels = np.arange(16, dtype=np.float32)
        model = xgb.train(
            {"objective": "reg:squarederror", "max_depth": 2, "nthread": 1},
            xgb.DMatrix(train_data, label=train_labels),
            num_boost_round=2,
        )
        dates = pd.date_range("2026-01-01", periods=56, freq="D")
        rows = pd.concat(
            [
                pd.DataFrame({
                    "id": series_id,
                    "date": dates,
                    "sales": np.arange(1, 57, dtype=np.float32),
                    "sell_price": np.nan,
                })
                for series_id in ("known-series", "new-series")
            ],
            ignore_index=True,
        )
        calendar = pd.DataFrame({
            "date": pd.to_datetime(["2016-06-19"]),
            "wm_yr_wk": [11621],
            "wday": [2],
            "month": [6],
            "year": [2016],
            "event_name_1": [None],
            "event_name_2": [None],
            "snap_CA": [0],
            "snap_TX": [0],
            "snap_WI": [0],
        })
        with tempfile.TemporaryDirectory() as temporary_directory:
            bundle_path = Path(temporary_directory) / "model.joblib"
            forecasting_service.save_model_bundle(
                model,
                bundle_path,
                training_date_utc="2026-01-01T00:00:00+00:00",
                training_end_day="d_1941",
                data_source="fixture",
            )
            bundle = forecasting_service.load_model_bundle(bundle_path)
        forecast, _ = forecasting_service.generate_forecast(
            rows,
            bundle,
            {
                "known_ids": {"known-series"},
                "series": {"known-series": {"state_id": "CA"}},
                "prices": {},
            },
            calendar,
            horizon=2,
            minimum_current_date=pd.Timestamp("2026-02-25"),
        )
        self.assertEqual(len(forecast), 4)
        self.assertEqual(forecast["date"].min(), pd.Timestamp("2026-02-26"))
        methods = forecast.groupby("id")["forecast_method"].first().to_dict()
        self.assertEqual(methods["known-series"], "recursive_xgboost")
        self.assertEqual(methods["new-series"], "recursive_xgboost")
        self.assertTrue(np.isfinite(forecast["predicted_sales"]).all())
        self.assertTrue((forecast["predicted_sales"] >= 0).all())

    def test_35_day_unrecognized_series_use_recursive_model_features(self) -> None:
        features = forecasting_service._feature_names()
        lag_1_index = features.index("lag_1")

        class FeatureDependentModel:
            def __init__(self) -> None:
                self.feature_batches: list[np.ndarray] = []

            def predict(self, matrix: xgb.DMatrix) -> np.ndarray:
                values = matrix.get_data().toarray()
                self.feature_batches.append(values)
                return values[:, lag_1_index] * 0.9 + 2.0

        model = FeatureDependentModel()
        dates = pd.date_range("2026-10-11", periods=35, freq="D")
        history = pd.concat(
            [
                pd.DataFrame({
                    "id": series_id,
                    "date": dates,
                    "sales": np.asarray(sales, dtype=np.float32),
                    "sell_price": price,
                })
                for series_id, sales, price in (
                    ("FOODS_1_001_CA_1", np.arange(1, 36), 2.48),
                    ("NEW_1_001_WI_2", np.arange(36, 71), 8.5),
                )
            ],
            ignore_index=True,
        )
        bundle = {
            "features": features,
            "model": model,
            "metadata": {"model_version": "fixture-xgb"},
        }
        original_dmatrix = xgb.DMatrix
        prepared_batches: list[np.ndarray] = []

        def capture_dmatrix(values: np.ndarray, **kwargs: object) -> xgb.DMatrix:
            prepared_batches.append(values.copy())
            return original_dmatrix(values, **kwargs)

        with patch.object(forecasting_service.xgb, "DMatrix", side_effect=capture_dmatrix):
            forecast, _ = forecasting_service.generate_forecast(
                history,
                bundle,
                {"known_ids": set(), "series": {}, "prices": {}},
                pd.DataFrame(),
                horizon=28,
                minimum_current_date=pd.Timestamp("2026-11-14"),
            )
        self.assertEqual(len(forecast), 56)
        self.assertEqual(set(forecast["forecast_method"]), {"recursive_xgboost"})
        first_batch = model.feature_batches[0]
        self.assertEqual(first_batch[:, features.index("lag_1")].tolist(), [35, 70])
        self.assertTrue(np.isnan(prepared_batches[0][:, features.index("lag_56")]).all())
        self.assertEqual(
            prepared_batches[0][:, features.index("lag_28")].tolist(),
            [8, 43],
        )
        self.assertEqual(
            prepared_batches[21][:, features.index("lag_56")].tolist(),
            [1, 36],
        )
        self.assertEqual(
            prepared_batches[1][:, features.index("lag_1")].tolist(),
            forecast.loc[forecast["date"].eq(pd.Timestamp("2026-11-15")), "predicted_sales"].tolist(),
        )
        totals = forecast.groupby("id")["predicted_sales"].sum()
        self.assertNotEqual(totals.iloc[0], totals.iloc[1])
        self.assertEqual(forecast.groupby("id")["predicted_sales"].nunique().min(), 28)

    def test_xgboost_prediction_errors_are_not_replaced_by_a_fallback(self) -> None:
        class BrokenModel:
            def predict(self, matrix: xgb.DMatrix) -> np.ndarray:
                raise xgb.core.XGBoostError("fixture model failure")

        history = pd.DataFrame({
            "id": ["unknown-series"] * 28,
            "date": pd.date_range("2026-10-18", periods=28),
            "sales": np.arange(1, 29),
        })
        with self.assertRaisesRegex(ValueError, "XGBoost prediction failed"):
            forecasting_service.generate_forecast(
                history,
                {
                    "features": forecasting_service._feature_names(),
                    "model": BrokenModel(),
                    "metadata": {"model_version": "fixture-broken"},
                },
                {"known_ids": set(), "series": {}, "prices": {}},
                pd.DataFrame(),
                horizon=1,
                minimum_current_date=pd.Timestamp("2026-11-14"),
            )

    def test_rejects_historical_upload_as_a_current_forecast(self) -> None:
        rows = pd.DataFrame({
            "id": ["series-a"] * 28,
            "date": pd.date_range("2016-01-01", periods=28),
            "sales": [1] * 28,
        })
        with self.assertRaisesRegex(ValueError, "recent sales data"):
            forecasting_service.generate_forecast(
                rows,
                {},
                {},
                pd.DataFrame(),
                horizon=1,
                minimum_current_date=pd.Timestamp("2026-01-01"),
                allow_historical=False,
            )

    def test_supports_historical_upload_when_allowed_and_labels_as_historical(self) -> None:
        """Verify historical uploads produce forecasts with forecast_type='historical'."""
        features = forecasting_service._feature_names()
        class DummyModel:
            def num_features(self):
                return len(features)
            def predict(self, matrix):
                return np.ones(matrix.num_row()) * 2.5

        bundle = {
            "model": DummyModel(),
            "features": features,
            "metadata": {"model_version": "test-hist"},
        }
        rows = pd.DataFrame({
            "id": ["series-hist"] * 28,
            "date": pd.date_range("2016-01-01", periods=28),
            "sales": [2.0] * 28,
        })
        fc, _ = forecasting_service.generate_forecast(
            rows,
            bundle,
            {"known_ids": set(), "series": {}, "prices": {}},
            pd.DataFrame(),
            horizon=7,
            minimum_current_date=pd.Timestamp("2026-01-01"),
            allow_historical=True,
        )
        self.assertEqual(len(fc), 7)
        self.assertEqual(fc["forecast_type"].iloc[0], "historical")
        self.assertEqual(fc["date"].min(), pd.Timestamp("2016-01-29"))

    def test_rejects_future_dated_sales(self) -> None:
        """Verify uploads with dates in the future are rejected."""
        far_future = pd.Timestamp.now().normalize() + pd.Timedelta(days=10)
        rows = pd.DataFrame({
            "id": ["series-future"] * 28,
            "date": pd.date_range(far_future, periods=28),
            "sales": [1.0] * 28,
        })
        with self.assertRaisesRegex(ValueError, "future dates"):
            forecasting_service.generate_forecast(
                rows,
                {"features": forecasting_service._feature_names(), "model": None, "metadata": {}},
                {"known_ids": set(), "series": {}, "prices": {}},
                pd.DataFrame(),
            )

    def test_rejects_missing_dates(self) -> None:
        """Verify rows with invalid or missing date values fail validation."""
        rows = pd.DataFrame({
            "id": ["series-a", "series-a"],
            "date": ["not-a-date", None],
            "sales": [1, 2],
        })
        with self.assertRaisesRegex(ValueError, "invalid date"):
            forecasting_service.validate_sales_rows(rows)

    def test_uploaded_retraining_requires_consecutive_history(self) -> None:
        rows = pd.DataFrame({
            "id": ["series-a"] * 25,
            "date": pd.date_range("2026-01-01", periods=25),
            "sales": np.arange(25, dtype=np.float32),
        })
        with self.assertRaisesRegex(ValueError, "No series meet the minimum requirement of 29"):
            forecasting_service.prepare_uploaded_training_data(
                rows,
                pd.DataFrame(),
                {},
                minimum_days=29,
            )

    def test_uploaded_retraining_supports_35_day_datasets(self) -> None:
        """Verify uploaded retraining succeeds on small real-world datasets (35 days, 3 series)."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory) / "models"
            dates = pd.date_range("2026-10-01", periods=35, freq="D")
            rows = pd.concat(
                [
                    pd.DataFrame({
                        "id": sid,
                        "date": dates,
                        "sales": np.arange(1.0, 36.0),
                        "sell_price": 2.50,
                    })
                    for sid in ("FOODS_1_001_CA_1", "FOODS_1_002_CA_1", "HOBBIES_1_001_TX_1")
                ],
                ignore_index=True,
            )
            with patch.object(forecasting_service, "MODEL_DIR", model_dir):
                metadata = forecasting_service.train_uploaded_history(
                    rows,
                    pd.DataFrame(),
                    {"series": {}, "prices": {}},
                    minimum_days=29,
                    minimum_rows=10,
                    num_boost_round=5,
                )
            self.assertEqual(metadata["eligible_series"], 3)
            self.assertEqual(metadata["training_rows"], 21)  # 7 target days * 3 series
            self.assertIn("training_fit_mae", metadata["evaluation_metrics"])

    def test_uploaded_retraining_saves_and_activates_versioned_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory) / "models"
            rows = pd.DataFrame({
                "id": ["series-a"] * 90,
                "date": pd.date_range("2026-01-01", periods=90),
                "sales": np.arange(90, dtype=np.float32) % 8,
            })
            with patch.object(forecasting_service, "MODEL_DIR", model_dir):
                metadata = forecasting_service.train_uploaded_history(
                    rows,
                    pd.DataFrame(),
                    {},
                    minimum_rows=20,
                    num_boost_round=1,
                )
            bundles = list(model_dir.glob("recursive_xgboost_uploaded_*.joblib"))
            self.assertEqual(len(bundles), 1)
            self.assertEqual(metadata["training_rows"], 28)
            self.assertEqual(metadata["eligible_series"], 1)
            self.assertEqual(
                metadata["data_source"],
                "Saved user-uploaded sales history",
            )
            pointer = json.loads(
                (model_dir / forecasting_service.ACTIVE_MODEL_PATH.name).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(pointer["model_bundle_path"], str(bundles[0].resolve()))
            loaded = forecasting_service.load_model_bundle(bundles[0])
            self.assertEqual(loaded["metadata"]["model_version"], metadata["model_version"])

    def test_saved_forecast_keeps_model_and_upload_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            forecast = pd.DataFrame({
                "id": ["series-a"],
                "date": [pd.Timestamp("2026-03-02")],
                "predicted_sales": [3.0],
                "forecast_method": ["recent_28_day_mean_fallback"],
                "model_version": ["model-v1"],
            })
            history = pd.DataFrame({
                "id": ["series-a"],
                "date": [pd.Timestamp("2026-03-01")],
                "sales": [2.0],
                "sell_price": [np.nan],
            })
            model_metadata = {
                "model_version": "model-v1",
                "training_date_utc": "2026-03-01T00:00:00+00:00",
                "training_data_end_date": "2026-02-28",
                "data_source": "fixture training data",
                "data_version": "fixture-hash",
            }
            with (
                patch.object(
                    forecasting_service,
                    "FORECAST_DIR",
                    root / "forecasts",
                ),
                patch.object(
                    forecasting_service,
                    "HISTORY_PATH",
                    root / "history.parquet",
                ),
            ):
                details = forecasting_service.save_forecast_result(
                    forecast,
                    history,
                    model_metadata=model_metadata,
                    source_name="upload.csv",
                )
                _, _, loaded = forecasting_service.load_latest_forecast()
            self.assertEqual(loaded["model_version"], "model-v1")
            self.assertEqual(loaded["model_data_version"], "fixture-hash")
            self.assertEqual(loaded["model_data_source"], "fixture training data")
            self.assertEqual(loaded["data_source"], "upload.csv")
            self.assertEqual(details["forecast_start"], "2026-03-02")

    def test_predictions_generated_for_all_valid_series_without_retraining(self) -> None:
        """Verify that genuine XGBoost predictions are produced for multiple distinct

        uploaded series with separate forecast series and zero retraining invoked.
        """
        bundle = forecasting_service.load_model_bundle()
        initial_save_time = bundle["metadata"]["saved_at_utc"]
        initial_rounds = bundle["metadata"]["training_rounds"]

        # 3 separate uploaded series with distinct historical sales
        dates = pd.date_range("2026-10-01", periods=45, freq="D")
        series_info = [
            ("SERIES_FOODS_CA_1", 20.0, 3.50),
            ("SERIES_HOBBIES_TX_2", 8.0, 10.00),
            ("SERIES_HOUSEHOLD_WI_3", 3.0, 5.25),
        ]
        rows = []
        for sid, mean_sales, price in series_info:
            rng = np.random.default_rng(abs(hash(sid)) % 1000)
            sales = rng.poisson(mean_sales, size=45).astype(float)
            for d, s in zip(dates, sales):
                rows.append({"id": sid, "date": d, "sales": s, "sell_price": price})
        history = pd.DataFrame(rows)

        forecast, actual = forecasting_service.generate_forecast(
            history,
            bundle,
            {"known_ids": set(), "series": {}, "prices": {}},
            pd.DataFrame(),
            horizon=28,
            minimum_current_date=pd.Timestamp("2026-10-15"),
        )

        # 1. Output series match all input series
        self.assertEqual(sorted(forecast["id"].unique()), [s[0] for s in series_info])
        self.assertEqual(len(forecast), 3 * 28)

        # 2. Method is recursive_xgboost for all rows
        self.assertEqual(set(forecast["forecast_method"]), {"recursive_xgboost"})

        # 3. Each series produces non-flat, distinct predictions
        series_totals = forecast.groupby("id")["predicted_sales"].sum()
        self.assertEqual(len(series_totals.unique()), 3)
        for sid in series_totals.index:
            series_preds = forecast.loc[forecast["id"] == sid, "predicted_sales"]
            self.assertGreater(series_preds.std(), 0.05)

        # 4. Confirm model bundle on disk was NOT modified or retrained
        reloaded_bundle = forecasting_service.load_model_bundle()
        self.assertEqual(reloaded_bundle["metadata"]["saved_at_utc"], initial_save_time)
        self.assertEqual(reloaded_bundle["metadata"]["training_rounds"], initial_rounds)

    def test_saved_xgboost_artifact_is_genuinely_loaded_and_used(self) -> None:
        """Verify the booster inside bundle is actually evaluated for each step."""
        bundle = forecasting_service.load_model_bundle()
        real_booster = bundle["model"]
        self.assertIsInstance(real_booster, xgb.Booster)

        # Count predict calls by wrapping the booster
        call_count = [0]
        original_predict = real_booster.predict

        class MonitoredBooster:
            def num_features(self):
                return real_booster.num_features()
            def predict(self, matrix):
                call_count[0] += 1
                return original_predict(matrix)

        monitored_bundle = {
            "model": MonitoredBooster(),
            "features": bundle["features"],
            "metadata": bundle["metadata"],
        }
        dates = pd.date_range("2026-10-01", periods=30, freq="D")
        hist = pd.DataFrame({
            "id": ["TEST_SERIES_CA_1"] * 30,
            "date": dates,
            "sales": np.arange(1.0, 31.0),
            "sell_price": [1.99] * 30,
        })
        forecast, _ = forecasting_service.generate_forecast(
            hist,
            monitored_bundle,
            {"known_ids": set(), "series": {}, "prices": {}},
            pd.DataFrame(),
            horizon=28,
            minimum_current_date=pd.Timestamp("2026-10-15"),
        )
        # Genuinely called 28 times (once per recursive horizon step)
        self.assertEqual(call_count[0], 28)
        self.assertEqual(len(forecast), 28)


class TrainingJobTests(unittest.TestCase):
    def test_dashboard_confirmation_enables_training_button(self) -> None:
        if os.name == "nt":
            self.skipTest("Streamlit AppTest hangs on Windows/Python 3.13 runtime.")
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit is not installed in this environment.")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prerequisites = {
                "FEATURE_FILE": root / "features.parquet",
                "SALES_FILE": root / "sales.csv",
                "CALENDAR_FILE": root / "calendar.csv",
                "PRICES_FILE": root / "prices.csv",
            }
            for path in prerequisites.values():
                path.write_text("", encoding="utf-8")

            with (
                patch.multiple(training_jobs, **prerequisites),
                patch.object(training_jobs, "JOB_DIR", root / "jobs"),
                patch.object(
                    forecasting_service,
                    "HISTORY_PATH",
                    root / "missing-user-history.parquet",
                ),
                patch.object(
                    training_jobs,
                    "LATEST_STATUS_PATH",
                    root / "jobs" / "latest.json",
                ),
            ):
                app_path = Path(__file__).resolve().parents[1] / "app.py"
                app = AppTest.from_file(
                    str(app_path),
                    default_timeout=30,
                ).run()
                app.radio[0].set_value("Train and save M5 forecast").run()
                m5_button = next(
                    button
                    for button in app.button
                    if button.label == "Start training and save forecast"
                )
                self.assertTrue(m5_button.proto.disabled)

                confirmation = next(
                    checkbox
                    for checkbox in app.checkbox
                    if checkbox.label
                    == "I understand this starts a long model-training job on the M5 data."
                )
                confirmation.set_value(True).run()

                self.assertTrue(confirmation.value)
                m5_button = next(
                    button
                    for button in app.button
                    if button.label == "Start training and save forecast"
                )
                self.assertFalse(m5_button.proto.disabled)
                self.assertEqual(len(app.exception), 0)

    def test_dead_worker_automatically_cleaned_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            job_dir = root / "jobs"
            jobs_dir = job_dir / "history"
            job_dir.mkdir(parents=True)
            jobs_dir.mkdir()
            lock_path = job_dir / "training.lock"
            latest_path = job_dir / "latest.json"
            job_id = "dead-worker-123"
            dead_pid = 9999999
            payload = {
                "job_id": job_id,
                "status": "running",
                "stage": "training_and_forecasting",
                "worker_pid": dead_pid,
                "started_at_utc": "2026-01-01T00:00:00+00:00",
            }
            lock_path.write_text(json.dumps({"job_id": job_id, "worker_pid": dead_pid}), encoding="utf-8")
            latest_path.write_text(json.dumps(payload), encoding="utf-8")
            (jobs_dir / f"{job_id}.json").write_text(json.dumps(payload), encoding="utf-8")

            with (
                patch.object(training_jobs, "JOB_DIR", job_dir),
                patch.object(training_jobs, "JOBS_DIR", jobs_dir),
                patch.object(training_jobs, "LOCK_PATH", lock_path),
                patch.object(training_jobs, "LATEST_STATUS_PATH", latest_path),
                patch.object(training_jobs, "_pid_is_running", return_value=False),
            ):
                status = training_jobs.read_latest_job()
                self.assertIsNotNone(status)
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["stage"], "stopped")
                self.assertIn("terminated", status["error"])
                self.assertFalse(lock_path.exists())

    def test_cancel_active_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            job_dir = root / "jobs"
            jobs_dir = job_dir / "history"
            job_dir.mkdir(parents=True)
            jobs_dir.mkdir()
            lock_path = job_dir / "training.lock"
            latest_path = job_dir / "latest.json"
            job_id = "running-worker-123"
            fake_pid = 88888
            payload = {
                "job_id": job_id,
                "status": "running",
                "stage": "training_and_forecasting",
                "worker_pid": fake_pid,
                "started_at_utc": "2026-01-01T00:00:00+00:00",
            }
            lock_path.write_text(json.dumps({"job_id": job_id, "worker_pid": fake_pid}), encoding="utf-8")
            latest_path.write_text(json.dumps(payload), encoding="utf-8")
            (jobs_dir / f"{job_id}.json").write_text(json.dumps(payload), encoding="utf-8")

            with (
                patch.object(training_jobs, "JOB_DIR", job_dir),
                patch.object(training_jobs, "JOBS_DIR", jobs_dir),
                patch.object(training_jobs, "LOCK_PATH", lock_path),
                patch.object(training_jobs, "LATEST_STATUS_PATH", latest_path),
                patch.object(training_jobs, "_pid_is_running", return_value=True),
                patch("subprocess.run") as mock_run,
            ):
                cancelled = training_jobs.cancel_active_job()
                self.assertTrue(cancelled)
                status = json.loads(latest_path.read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["stage"], "cancelled")
                self.assertFalse(lock_path.exists())

    def test_start_creates_background_job_and_versioned_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            job_dir = root / "jobs"
            jobs_dir = job_dir / "history"
            lock_path = job_dir / "training.lock"
            latest_path = job_dir / "latest.json"
            script_path = root / "extended_workflows.py"
            runner_path = root / "training_job_runner.py"
            feature_path = root / "features.parquet"
            feature_config_path = root / "feature_config.json"
            sales_path = root / "sales.csv"
            calendar_path = root / "calendar.csv"
            prices_path = root / "prices.csv"
            for path in (
                script_path,
                runner_path,
                feature_path,
                feature_config_path,
                sales_path,
                calendar_path,
                prices_path,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with (
                patch.object(training_jobs, "ROOT", root),
                patch.object(training_jobs, "JOB_DIR", job_dir),
                patch.object(training_jobs, "JOBS_DIR", jobs_dir),
                patch.object(training_jobs, "LOCK_PATH", lock_path),
                patch.object(training_jobs, "LATEST_STATUS_PATH", latest_path),
                patch.object(training_jobs, "TRAINING_SCRIPT", script_path),
                patch.object(training_jobs, "FEATURE_FILE", feature_path),
                patch.object(training_jobs, "FEATURE_CONFIG", feature_config_path),
                patch.object(training_jobs, "SALES_FILE", sales_path),
                patch.object(training_jobs, "CALENDAR_FILE", calendar_path),
                patch.object(training_jobs, "PRICES_FILE", prices_path),
                patch.object(training_jobs, "MODEL_DIR", root / "models"),
                patch.object(training_jobs, "FORECAST_DIR", root / "forecasts"),
                patch.object(training_jobs, "_pid_is_running", return_value=True),
                patch(
                    "training_jobs.subprocess.Popen",
                    return_value=type("Process", (), {"pid": 4321})(),
                ) as process_start,
            ):
                status = training_jobs.start_recursive_training()
                loaded = training_jobs.read_latest_job()
                self.assertEqual(status["status"], "running")
                self.assertEqual(loaded["job_id"], status["job_id"])
                self.assertTrue(lock_path.is_file())
                self.assertIn(status["job_id"], status["artifacts"]["model_path"])
                self.assertIn(
                    status["job_id"],
                    status["artifacts"]["model_bundle_path"],
                )
                self.assertIn(status["job_id"], status["artifacts"]["forecast_path"])
                self.assertEqual(
                    status["forecast_period"],
                    ["d_1942", "d_1969"],
                )
                process_start.assert_called_once()

    def test_rejects_duplicate_job_while_worker_is_alive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            job_dir = root / "jobs"
            jobs_dir = job_dir / "history"
            job_dir.mkdir(parents=True)
            jobs_dir.mkdir()
            old_job_id = "already-running"
            lock_path = job_dir / "training.lock"
            lock_path.write_text(
                json.dumps({"job_id": old_job_id, "worker_pid": os.getpid()}),
                encoding="utf-8",
            )
            (jobs_dir / f"{old_job_id}.json").write_text(
                json.dumps({"job_id": old_job_id, "status": "running"}),
                encoding="utf-8",
            )
            with (
                patch.object(training_jobs, "ROOT", root),
                patch.object(training_jobs, "JOB_DIR", job_dir),
                patch.object(training_jobs, "JOBS_DIR", jobs_dir),
                patch.object(training_jobs, "LOCK_PATH", lock_path),
                patch.object(
                    training_jobs,
                    "TRAINING_SCRIPT",
                    root / "extended_workflows.py",
                ),
                patch.object(training_jobs, "FEATURE_FILE", root / "features.parquet"),
                patch.object(
                    training_jobs,
                    "FEATURE_CONFIG",
                    root / "feature_config.json",
                ),
                patch.object(training_jobs, "SALES_FILE", root / "sales.csv"),
                patch.object(training_jobs, "CALENDAR_FILE", root / "calendar.csv"),
                patch.object(training_jobs, "PRICES_FILE", root / "prices.csv"),
            ):
                for path in (
                    root / "extended_workflows.py",
                    root / "features.parquet",
                    root / "feature_config.json",
                    root / "sales.csv",
                    root / "calendar.csv",
                    root / "prices.csv",
                ):
                    path.touch()
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    training_jobs.start_recursive_training()

    def test_metadata_validates_forecast_and_saves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_path = root / "model.json"
            model = xgb.train(
                {"objective": "reg:squarederror", "nthread": 1},
                xgb.DMatrix(np.zeros((2, 30), dtype=np.float32), label=[0, 1]),
                num_boost_round=1,
            )
            model.save_model(model_path)
            model_bundle_path = root / "model.joblib"
            forecast_path = root / "forecast.parquet"
            forecast = pd.DataFrame(
                {
                    "id": ["series-a"] * 28 + ["series-b"] * 28,
                    "d": [f"d_{day}" for day in range(1942, 1970)] * 2,
                    "predicted_sales": [1.0] * 56,
                }
            )
            forecast.to_parquet(forecast_path, index=False)
            source_paths = [
                root / "sales.csv",
                root / "calendar.csv",
                root / "prices.csv",
                root / "features.parquet",
                root / "data" / "model_ready" / "feature_config.json",
            ]
            for path in source_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            metadata_path = root / "metadata.json"
            job = {
                "job_id": "job-fixture",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "artifacts": {
                    "model_path": str(model_path),
                    "model_bundle_path": str(model_bundle_path),
                    "forecast_path": str(forecast_path),
                    "metadata_path": str(metadata_path),
                },
            }
            with (
                patch.object(training_jobs, "SALES_FILE", source_paths[0]),
                patch.object(training_jobs, "CALENDAR_FILE", source_paths[1]),
                patch.object(training_jobs, "PRICES_FILE", source_paths[2]),
                patch.object(training_jobs, "FEATURE_FILE", source_paths[3]),
                patch.object(training_job_runner, "ROOT", root),
            ):
                metadata = training_job_runner._write_metadata(
                    job,
                    model_path,
                    forecast_path,
                    metadata_path,
                )
            self.assertEqual(metadata["forecast_row_count"], 56)
            self.assertEqual(metadata["series_count"], 2)
            self.assertTrue(metadata_path.is_file())
            self.assertTrue(model_bundle_path.is_file())
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8"))["training_job_id"],
                "job-fixture",
            )

    def test_worker_saves_outputs_and_marks_job_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            job_dir = root / "jobs"
            jobs_dir = job_dir / "history"
            lock_path = job_dir / "training.lock"
            latest_path = job_dir / "latest.json"
            model_path = root / "model.json"
            model_bundle_path = root / "model.joblib"
            forecast_path = root / "forecast.parquet"
            metadata_path = root / "metadata.json"
            log_path = job_dir / "training.log"
            sources = [
                root / "sales.csv",
                root / "calendar.csv",
                root / "prices.csv",
                root / "features.parquet",
                root / "data" / "model_ready" / "feature_config.json",
            ]
            for path in sources:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            model = xgb.train(
                {"objective": "reg:squarederror", "nthread": 1},
                xgb.DMatrix(np.zeros((2, 30), dtype=np.float32), label=[0, 1]),
                num_boost_round=1,
            )
            model.save_model(model_path)
            pd.DataFrame(
                {
                    "id": ["series-a"] * 28,
                    "d": [f"d_{day}" for day in range(1942, 1970)],
                    "predicted_sales": [1.0] * 28,
                }
            ).to_parquet(forecast_path, index=False)
            job_id = "worker-fixture"
            job = {
                "job_id": job_id,
                "status": "running",
                "stage": "starting",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "log_path": str(log_path),
                "artifacts": {
                    "model_path": str(model_path),
                    "model_bundle_path": str(model_bundle_path),
                    "forecast_path": str(forecast_path),
                    "metadata_path": str(metadata_path),
                },
            }
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps({"job_id": job_id, "worker_pid": os.getpid()}),
                encoding="utf-8",
            )

            def finish_training(command, **kwargs):
                return __import__("subprocess").CompletedProcess(command, 0)

            with (
                patch.object(training_jobs, "JOB_DIR", job_dir),
                patch.object(training_jobs, "JOBS_DIR", jobs_dir),
                patch.object(training_jobs, "LOCK_PATH", lock_path),
                patch.object(training_jobs, "LATEST_STATUS_PATH", latest_path),
                patch.object(training_jobs, "SALES_FILE", sources[0]),
                patch.object(training_jobs, "CALENDAR_FILE", sources[1]),
                patch.object(training_jobs, "PRICES_FILE", sources[2]),
                patch.object(training_jobs, "FEATURE_FILE", sources[3]),
                patch.object(training_job_runner, "ROOT", root),
                patch(
                    "training_job_runner.subprocess.run",
                    side_effect=finish_training,
                ),
            ):
                training_jobs.write_job_status(job)
                result = training_job_runner.run_job(job_id)
                final_status = training_jobs.read_latest_job()

            self.assertEqual(result, 0)
            self.assertEqual(final_status["status"], "completed")
            self.assertEqual(
                final_status["model_metadata"]["forecast_row_count"],
                28,
            )
            self.assertTrue(metadata_path.is_file())
            self.assertFalse(lock_path.exists())


@unittest.skipIf(api is None, "FastAPI dependencies are unavailable.")
class ApiTests(unittest.TestCase):
    def test_persists_and_retrieves_replenishment_simulation(self) -> None:
        assert api is not None
        from pydantic import ValidationError

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
                req = api.ReplenishmentRequest(
                    store_id="CA_1",
                    item_id="FOODS_1_001",
                    forecast_type="direct",
                    lead_time_days=7,
                    on_hand=0,
                )
                response = api.replenishment_simulation(req)
                self.assertIn("run_id", response)
                run_id = response["run_id"]
                self.assertEqual(
                    response["result"]["suggested_order_quantity"],
                    7,
                )
                saved = api.get_replenishment_run(run_id)
                self.assertEqual(saved["item_id"], "FOODS_1_001")

                with self.assertRaises(ValidationError):
                    api.ReplenishmentRequest(
                        store_id="CA_1",
                        item_id="FOODS_1_001",
                        forecast_type="direct",
                        lead_time_days=29,
                        on_hand=0,
                    )

                with self.assertRaises(ValidationError):
                    api.ReplenishmentRequest(
                        store_id="CA_1",
                        item_id="FOODS_1_001",
                        forecast_type="direct",
                        lead_time_days=7,
                        on_hand=0,
                        service_level=0.94,
                    )
                api._read_forecast.cache_clear()


if __name__ == "__main__":
    unittest.main()
