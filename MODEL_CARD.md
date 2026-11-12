# Model card: Store-level demand forecast

## Intended use

This project demonstrates daily unit-sales forecasting for the item/store series in
the M5 Forecasting Accuracy dataset. It is an offline research and planning example,
not a live retail replenishment system.

## Model choices

The existing production artifact is a recursive XGBoost regressor. The recorded
comparison reports recursive MAE of 1.1113 for XGBoost, 1.1453 for the two-stage
approach, and 1.1669 for LightGBM. XGBoost was retained because it had the lowest
recursive MAE in that comparison and uses one prediction model in the recursive
forecast loop.

The comparison does not establish that XGBoost is universally better. In normal
validation, the two-stage model had the lowest MAE. It also does not compare the
algorithms under every possible feature, training, or forecast design.

The PRD calls for direct forecasts at horizons 1–28. `direct_forecasting.py`
implements a separate horizon-aware XGBoost candidate for this serving design. It
does not replace or silently reinterpret the saved recursive results. Three
matched folds were trained. Direct MAE was 1.0666, 1.0587, and 1.0792, compared
with recursive XGBoost MAE of 1.0401, 1.0383, and 1.0638. Direct XGBoost also
had higher RMSE and WAPE than both recursive XGBoost and the recent-average
baseline in each fold. It is not selected to replace the recursive candidate.
A comparison with direct LightGBM under identical folds has not been performed.

## Data and validation

- Source: M5 `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv`.
- The existing recursive model was selected using historical validation evidence
  already recorded in `model.ipynb`.
- The saved rolling comparison uses three recursive 28-day folds and a fixed
  recent-average baseline.
- `d_1914`–`d_1941` has already been used for model selection and reporting. It is
  not an untouched final test period.
- The direct-model evaluation command can score its `d_1914`–`d_1941` forecast,
  but those results must be described as a reused period. A new later holdout or
  additional walk-forward folds are needed for an unbiased final claim.
- Direct walk-forward model versions were `8b39cab39a60`, `11a813581abf`, and
  `66f05a514c2e`; all used data version
  `57e4fa85e3b77600a0e9166a23a51fa8e88c6c162fbddf2802d5b8c99bf72689`.

## Features and forecast behavior

The direct candidate uses 56 days of sales history at a forecast origin, calendar
features for the requested target day, state SNAP availability, series identity
codes, and price features known at the origin. Origin-known price features are
carried through the 28-day horizon; future realized prices are not used. Its
training rows are origin/horizon examples, and the horizon is an explicit feature.
Predictions are clipped at zero.
The existing notebook's `price_norm` is based on a full-series median, which
includes future dates relative to earlier training origins; the direct candidate
omits that feature rather than leak that future information.

The recursive and direct candidates are different forecasting designs. Their
scores should only be compared on the same origin, horizon, data version, series
population, and evaluation metrics.

## Limitations and risks

- M5 records sales, not confirmed unconstrained demand; inventory, availability,
  and stock-out labels are not available.
- The synthetic replenishment simulator is not an ordering recommendation. Its
  output depends on user-provided inventory and variability assumptions and must
  be reviewed by a human.
- Future forecasts for `d_1942`–`d_1969` have no actuals in the supplied dataset
  and cannot be scored here.
- This project has not established current real-world demand accuracy, operational
  service levels, or performance on another retailer's data.

## Reproducibility

Training records the model hash, data hash, feature-schema hash, training cutoff,
and training configuration in `models/direct_xgboost_metadata.json`. The model
and data are large generated artifacts and are not required to be committed with
the source code.

Train the direct candidate using examples that do not include the existing
28-day holdout targets:

```powershell
python .\direct_forecasting.py train --training-end-day 1913
python .\direct_forecasting.py forecast --origin-day 1913
python .\direct_forecasting.py evaluate --forecast .\data\model_ready\direct_forecast_d1914_d1941.parquet
```

Training is computationally intensive because it creates millions of
series/origin/horizon rows (tens of millions at the default cutoff) and fits
150 boosting rounds by default. Safety-stock calculations also assume independent
daily forecast errors and use a user-supplied error standard deviation.
