"""Comprehensive unit tests for inventory replenishment logic and edge cases."""

import math
import unittest
from inventory import SERVICE_Z, ReplenishmentInputs, simulate_replenishment


class InventoryExtendedTests(unittest.TestCase):
    def test_exact_manual_calculation(self):
        # Forecast: 10 units/day for 14 days
        forecasts = [10.0] * 14
        # Lead time = 7 days, review period = 3 days (protection period = 10 days)
        # on_hand = 40, on_order = 20, backorders = 5 -> inventory_position = 55
        # service_level = 0.95 -> z = 1.65
        # daily_residual_std = 2.0
        # lead_time_demand = 70.0
        # protection_period_demand = 100.0
        # lead_time_safety_stock = 1.65 * sqrt(7) * 2.0 = 8.730979
        # protection_safety_stock = 1.65 * sqrt(10) * 2.0 = 10.435516
        # target_stock = 100.0 + 10.435516 = 110.435516
        # raw_order_quantity = 110.435516 - 55 = 55.435516
        # pack_size = 12 -> ceil(55.435516 / 12) * 12 = 5 * 12 = 60
        inputs = ReplenishmentInputs(
            on_hand=40.0,
            on_order=20.0,
            backorders=5.0,
            lead_time_days=7,
            review_period_days=3,
            pack_size=12,
            minimum_order_quantity=0,
            service_level=0.95,
            daily_residual_std=2.0,
        )
        result = simulate_replenishment(forecasts, inputs)
        self.assertAlmostEqual(result["inventory_position"], 55.0)
        self.assertAlmostEqual(result["lead_time_demand"], 70.0)
        self.assertAlmostEqual(result["protection_period_demand"], 100.0)
        expected_ss = 1.65 * math.sqrt(10) * 2.0
        self.assertAlmostEqual(result["protection_period_safety_stock"], expected_ss, places=4)
        self.assertAlmostEqual(result["target_stock"], 100.0 + expected_ss, places=4)
        self.assertEqual(result["suggested_order_quantity"], 60)

    def test_moq_enforcement(self):
        forecasts = [2.0] * 7
        inputs = ReplenishmentInputs(
            on_hand=5.0,
            on_order=0.0,
            backorders=0.0,
            lead_time_days=3,
            review_period_days=0,
            pack_size=1,
            minimum_order_quantity=50,
            service_level=0.95,
            daily_residual_std=0.0,
        )
        # Demand = 6, inventory_pos = 5, raw_order = 1, MOQ = 50 -> suggested = 50
        result = simulate_replenishment(forecasts, inputs)
        self.assertEqual(result["suggested_order_quantity"], 50)

    def test_zero_order_when_inventory_sufficient(self):
        forecasts = [5.0] * 7
        inputs = ReplenishmentInputs(
            on_hand=100.0,
            on_order=0.0,
            backorders=0.0,
            lead_time_days=3,
            review_period_days=0,
            pack_size=5,
            minimum_order_quantity=20,
            service_level=0.95,
            daily_residual_std=0.0,
        )
        result = simulate_replenishment(forecasts, inputs)
        self.assertEqual(result["suggested_order_quantity"], 0)
        self.assertEqual(result["raw_order_quantity"], 0.0)
        self.assertEqual(result["severity"], "No simulated reorder trigger")

    def test_reorder_trigger_when_at_or_below_rop(self):
        forecasts = [10.0] * 7
        inputs = ReplenishmentInputs(
            on_hand=30.0,
            on_order=0.0,
            backorders=0.0,
            lead_time_days=3,
            review_period_days=0,
            service_level=0.95,
            daily_residual_std=0.0,
        )
        # ROP = 30.0, inventory_pos = 30.0 -> triggers review
        result = simulate_replenishment(forecasts, inputs)
        self.assertIn("at or below reorder point", result["severity"])

    def test_validation_errors(self):
        valid_f = [5.0] * 10
        # Empty forecasts
        with self.assertRaises(ValueError):
            simulate_replenishment([], ReplenishmentInputs(10, 0, 0, 1))

        # Negative on_hand
        with self.assertRaises(ValueError):
            simulate_replenishment(valid_f, ReplenishmentInputs(-1, 0, 0, 1))

        # Negative lead time
        with self.assertRaises(ValueError):
            simulate_replenishment(valid_f, ReplenishmentInputs(10, 0, 0, 0))

        # Invalid service level
        with self.assertRaises(ValueError):
            simulate_replenishment(valid_f, ReplenishmentInputs(10, 0, 0, 1, service_level=0.75))

        # Negative demand in forecasts
        with self.assertRaises(ValueError):
            simulate_replenishment([-1.0, 5.0], ReplenishmentInputs(10, 0, 0, 1))

        # Horizon shorter than protection period
        with self.assertRaises(ValueError):
            simulate_replenishment([5.0] * 3, ReplenishmentInputs(10, 0, 0, lead_time_days=2, review_period_days=2))


if __name__ == "__main__":
    unittest.main()
