"""
tests/test_simulator.py
Unit tests for the inventory replenishment logic.
"""
from src.replenishment.simulator import simulate_replenishment

def test_replenishment_high_stock():
    # 28 days of constant 10 units demand (Total = 280)
    fcst = [10.0] * 28
    
    # We have 500 units in stock. No stockout expected.
    res = simulate_replenishment(fcst, current_stock=500, lead_time_days=7, service_level=0.95)
    
    assert res['avg_daily_demand'] == 10.0
    assert res['std_daily_demand'] == 0.0
    assert res['total_forecast_demand'] == 280.0
    assert res['safety_stock'] == 0.0 # No std dev = no safety buffer needed
    assert res['reorder_point'] == 70.0 # lead time (7) * avg (10)
    assert res['suggested_order_qty'] == 0.0 # We have way more than we need
    assert res['stockout_risk'] == '🟢 LOW'

def test_replenishment_low_stock():
    # 28 days of variable demand around 10
    fcst = [10, 12, 8, 10, 15, 5, 10] * 4 # avg 10, std dev > 0
    
    # We have 20 units in stock. Huge stockout risk.
    res = simulate_replenishment(fcst, current_stock=20, lead_time_days=7, service_level=0.99)
    
    assert res['total_forecast_demand'] == 280.0
    assert res['stockout_risk'] == '🔴 HIGH'
    assert res['projected_ending_stock'] == -260.0
    assert res['suggested_order_qty'] > 260.0 # Must order the deficit plus safety stock
