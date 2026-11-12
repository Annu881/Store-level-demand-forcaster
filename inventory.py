"""Transparent inventory replenishment calculations for simulated scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite, sqrt
from typing import Sequence


SERVICE_Z = {
    0.80: 0.84,
    0.90: 1.28,
    0.95: 1.65,
    0.99: 2.33,
}
DISCLAIMER = "Simulated, not operational. Human review required."


@dataclass(frozen=True)
class ReplenishmentInputs:
    on_hand: float
    on_order: float
    backorders: float
    lead_time_days: int
    review_period_days: int = 0
    pack_size: int = 1
    minimum_order_quantity: int = 0
    service_level: float = 0.95
    daily_residual_std: float = 0.0


def simulate_replenishment(
    forecasts: Sequence[float],
    inputs: ReplenishmentInputs,
) -> dict[str, float | int | str]:
    if len(forecasts) == 0:
        raise ValueError("At least one daily forecast is required.")
    if inputs.lead_time_days < 1:
        raise ValueError("lead_time_days must be at least 1.")
    if inputs.review_period_days < 0:
        raise ValueError("review_period_days cannot be negative.")
    if inputs.pack_size < 1:
        raise ValueError("pack_size must be at least 1.")
    if inputs.minimum_order_quantity < 0:
        raise ValueError("minimum_order_quantity cannot be negative.")
    quantities = (
        inputs.on_hand,
        inputs.on_order,
        inputs.backorders,
        inputs.daily_residual_std,
    )
    if any(not isfinite(value) for value in quantities):
        raise ValueError("Inventory inputs must be finite numbers.")
    if any(value < 0 for value in (inputs.on_hand, inputs.on_order, inputs.backorders)):
        raise ValueError("Inventory quantities cannot be negative.")
    if inputs.daily_residual_std < 0:
        raise ValueError("daily_residual_std cannot be negative.")
    if inputs.service_level not in SERVICE_Z:
        raise ValueError(f"service_level must be one of {sorted(SERVICE_Z)}.")
    forecast_values = [float(value) for value in forecasts]
    if any(not isfinite(value) for value in forecast_values):
        raise ValueError("Forecast demand must contain only finite numbers.")
    if any(value < 0 for value in forecast_values):
        raise ValueError("Forecast demand cannot contain negative values.")

    if inputs.lead_time_days > len(forecast_values):
        raise ValueError(
            "The forecast horizon does not cover the configured lead time."
        )
    protection_days = inputs.lead_time_days + inputs.review_period_days
    if protection_days > len(forecast_values):
        raise ValueError(
            "The forecast horizon does not cover lead time plus review period."
        )

    lead_time_demand = sum(forecast_values[: inputs.lead_time_days])
    protection_period_demand = sum(forecast_values[:protection_days])
    z_score = SERVICE_Z[inputs.service_level]
    lead_time_safety_stock = (
        z_score * sqrt(inputs.lead_time_days) * inputs.daily_residual_std
    )
    protection_safety_stock = (
        z_score * sqrt(protection_days) * inputs.daily_residual_std
    )
    inventory_position = inputs.on_hand + inputs.on_order - inputs.backorders
    reorder_point = lead_time_demand + lead_time_safety_stock
    target_stock = protection_period_demand + protection_safety_stock
    raw_order_quantity = max(0.0, target_stock - inventory_position)
    if raw_order_quantity == 0:
        suggested_order = 0
    else:
        minimum_adjusted = max(
            raw_order_quantity,
            float(inputs.minimum_order_quantity),
        )
        suggested_order = int(ceil(minimum_adjusted / inputs.pack_size) * inputs.pack_size)

    if inventory_position <= reorder_point:
        severity = "Review: inventory position is at or below reorder point"
    else:
        severity = "No simulated reorder trigger"

    return {
        "inventory_position": inventory_position,
        "lead_time_demand": lead_time_demand,
        "protection_period_demand": protection_period_demand,
        "lead_time_safety_stock": lead_time_safety_stock,
        "protection_period_safety_stock": protection_safety_stock,
        "safety_stock": protection_safety_stock,
        "reorder_point": reorder_point,
        "target_stock": target_stock,
        "raw_order_quantity": raw_order_quantity,
        "suggested_order_quantity": suggested_order,
        "service_level": inputs.service_level,
        "severity": severity,
        "disclaimer": DISCLAIMER,
    }


def inputs_as_dict(inputs: ReplenishmentInputs) -> dict[str, float | int]:
    return asdict(inputs)
