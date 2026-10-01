"""
src/replenishment/simulator.py
Inventory replenishment simulation using LightGBM demand forecasts.
Calculates: Safety Stock, Reorder Point, and Suggested Order Quantity.
This is a DECISION SUPPORT tool — not an autonomous ordering system.
"""
import numpy as np


def simulate_replenishment(
    forecast_units: list,
    current_stock: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    holding_cost_per_unit: float = 0.5,
    stockout_cost_per_unit: float = 5.0,
) -> dict:
    """
    Given a list of daily demand forecasts, computes inventory replenishment metrics.

    Args:
        forecast_units     : list of daily predicted demand (from LightGBM)
        current_stock      : current on-hand inventory in units
        lead_time_days     : number of days it takes to receive an order
        service_level      : desired probability of no stockout (e.g. 0.95 = 95%)
        holding_cost_per_unit : $ cost to hold 1 unit for the forecast period
        stockout_cost_per_unit: $ cost of losing 1 unit to a stockout

    Returns:
        dict with all replenishment KPIs
    """
    forecast = np.array(forecast_units, dtype=float)
    n_days   = len(forecast)

    # ── Demand statistics ──────────────────────────────────────────────────────
    avg_daily_demand  = float(np.mean(forecast))
    std_daily_demand  = float(np.std(forecast))
    total_demand      = float(np.sum(forecast))

    # ── Lead-time demand ───────────────────────────────────────────────────────
    lead_time_demand  = avg_daily_demand * lead_time_days

    # ── Safety Stock (z-score method) ─────────────────────────────────────────
    # z-score based on service level (one-tailed normal distribution)
    z_score_map = {0.90: 1.28, 0.95: 1.645, 0.98: 2.054, 0.99: 2.326}
    z = z_score_map.get(service_level, 1.645)
    safety_stock = z * std_daily_demand * np.sqrt(lead_time_days)

    # ── Reorder Point ─────────────────────────────────────────────────────────
    reorder_point = lead_time_demand + safety_stock

    # ── Suggested Order Quantity (simple EOQ-inspired) ────────────────────────
    # Order enough to cover full forecast + safety stock − current stock
    suggested_order = max(0.0, total_demand + safety_stock - current_stock)

    # ── Stockout risk ─────────────────────────────────────────────────────────
    days_of_stock           = current_stock / avg_daily_demand if avg_daily_demand > 0 else n_days
    projected_ending_stock  = current_stock - total_demand
    stockout_risk           = "🔴 HIGH" if projected_ending_stock < 0 else (
                              "🟡 MEDIUM" if projected_ending_stock < safety_stock else "🟢 LOW")

    # ── Cost estimates ────────────────────────────────────────────────────────
    holding_cost  = max(0.0, projected_ending_stock) * holding_cost_per_unit
    stockout_cost = abs(min(0.0, projected_ending_stock)) * stockout_cost_per_unit

    return {
        # Demand stats
        "avg_daily_demand":       round(avg_daily_demand, 2),
        "std_daily_demand":       round(std_daily_demand, 2),
        "total_forecast_demand":  round(total_demand, 0),
        "forecast_days":          n_days,

        # Inventory params
        "lead_time_days":         lead_time_days,
        "service_level_pct":      f"{int(service_level*100)}%",
        "z_score":                z,

        # Replenishment outputs
        "safety_stock":           round(safety_stock, 1),
        "reorder_point":          round(reorder_point, 1),
        "suggested_order_qty":    round(suggested_order, 0),
        "days_of_stock_remaining":round(days_of_stock, 1),

        # Risk & cost
        "projected_ending_stock": round(projected_ending_stock, 0),
        "stockout_risk":          stockout_risk,
        "estimated_holding_cost": round(holding_cost, 2),
        "estimated_stockout_cost":round(stockout_cost, 2),
    }


if __name__ == "__main__":
    # Quick test
    sample_forecast = [3.2, 4.1, 2.8, 5.0, 3.5, 4.8, 2.9] * 4  # 28 days
    result = simulate_replenishment(
        forecast_units=sample_forecast,
        current_stock=80,
        lead_time_days=7,
        service_level=0.95
    )
    print("\n=== REPLENISHMENT SIMULATION ===")
    for k, v in result.items():
        print(f"  {k:<30}: {v}")
