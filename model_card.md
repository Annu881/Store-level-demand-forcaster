# Model Card: Store-Level Demand Forecaster

**Intended use:** Educational retail demand forecasting, analyst decision support, and demonstration of an end-to-end ML lifecycle.

**Not intended for:** Autonomous ordering, supplier communication, confirmed stock-out detection, guaranteed demand or availability claims, or real operational deployment without data integration, governance, validation, and human approval.

**Data and validation:** 
- Uses M5 retail sales, calendar/events/SNAP information, and weekly price data.
- Validation uses rolling-origin walk-forward evaluation plus one final untouched 28-day holdout, with baseline comparison and segmented metrics.

**Limitations:** 
- The model predicts observed sales rather than unconstrained demand.
- M5 lacks inventory and availability fields. Potential censoring is inferred only. 
- Forecasts are not guarantees. Price features depend on an explicit known-at-origin assumption. 
- Feature importance reflects predictive association, not causality.

**Traceability:** 
Each result records data version, model version, feature-schema version, forecast origin, training configuration, metrics, horizon, and generation timestamp.
