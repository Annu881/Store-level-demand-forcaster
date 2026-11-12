# Store-Level Demand Forecaster

This project explores and forecasts daily retail unit sales for item/store series using the M5 Forecasting dataset. It includes notebook-based experiments and a recursive XGBoost forecast, plus a separate horizon-aware direct XGBoost candidate, data-quality checks, a FastAPI service, and a Streamlit dashboard.

## Project files

```text
.
├── EDA.ipynb                 # Dataset exploration, feature engineering, and model-ready data
├── model.ipynb               # Baselines, model experiments, training, and forecasting
├── direct_forecasting.py     # Direct 1–28-day XGBoost training, forecasting, evaluation
├── data_quality.py           # M5 schema, key, date, and value checks
├── inventory.py              # Validated replenishment scenario calculations
├── api.py                    # FastAPI forecasts, analytics, and simulation endpoints
├── MODEL_CARD.md             # Intended use, evidence, and limitations
├── tests/                    # Focused automated tests
├── data/
│   ├── raw/                  # M5 source CSV files and intermediate training features
│   └── model_ready/          # Engineered features, feature configuration, forecast output
└── models/
    ├── final_xgboost.json    # Trained XGBoost model used for forecasting
    └── xgb_cache/            # XGBoost external-memory training cache
```

The M5 competition's raw sales and price data and large engineered feature tables are not stored in Git. Download `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv` from the [Kaggle M5 Forecasting Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place them in `data/raw/` to rerun feature preparation or model training. The small calendar file and the saved forecast outputs used by the dashboard are included.

## Quick start: run the existing frontend

Open PowerShell in the project folder (the inner `Store-level-demand-forcaster` directory):

```powershell
cd C:\Users\acer.DESKTOP-F8EJN0N\Store_level_demand\Store-level-demand-forcaster
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

Open the `Local URL` printed by Streamlit, usually `http://localhost:8501`. Stop the app with **Ctrl+C**.

This starts the frontend using the **saved M5 forecasts already in `data/model_ready/`**. It does not train a model, create forecasts, or call the API. In the sidebar, try **Explore sales estimates**, **Past accuracy**, **Replenishment simulator**, **Data quality**, and **About this tool**. If an optional generated forecast file is absent, it will not be offered.

The saved forecast the dashboard shows is determined by the forecast Parquet files and paths configured in `app.py`. The `d_1942`–`d_1969` projection is an example based on the historical M5 dataset, not a forecast for the current date or for new Walmart data.

### How the pieces connect

```text
M5 sales_train_evaluation.csv + calendar.csv + sell_prices.csv
                         |
                         v
EDA.ipynb  ->  feature preparation in data/model_ready/
                         |
                         v
model.ipynb  ->  trains/evaluates the selected recursive XGBoost model
                         |
                         v
extended_workflows.py  ->  optional rolling tests or future M5 forecast
                         |
                         v
saved forecast Parquet files  ->  app.py reads and displays results
```

`model.ipynb` is the original experiment and recursive-model training workflow. `models/final_xgboost.json` is its model artifact. Opening the dashboard only reads saved forecasts; it does not automatically load this model or retrain it. After changing data or training a new model, generate and save a matching forecast, then configure the dashboard to display that forecast.

## Workflow

### 1. Explore and prepare the data

Run `EDA.ipynb` from top to bottom. It loads the sales, calendar, and sell-price CSVs and explores sales patterns, missing values, price and event effects, and demand behavior. It then:

- Prepares the daily item/store sales data and joins calendar and price information.
- Engineers calendar, price, lag, rolling-window, and demand-related features.
- Examines feature availability and uses a time-based 28-day holdout.
- Saves the engineered modeling dataset and feature configuration under `data/model_ready/`.

The notebook expects these source files in `data/raw/`:

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

### 2. Train and forecast

Run `model.ipynb` from top to bottom after preparing the data. It compares moving-average baselines with LightGBM, XGBoost, and SARIMAX experiments, evaluates validation performance, and inspects XGBoost feature importance. The final XGBoost training uses batched Parquet input and an external-memory cache to handle the large feature dataset.

The final forecasting section predicts days `d_1914`–`d_1941` (28 days). It updates lag and rolling features with each day's predictions, rather than using actual future sales as inputs. It evaluates predictions against the available actuals and saves them to:

```text
data/model_ready/final_forecast_d1914_d1941.parquet
```

The forecasting section loads `models/final_xgboost.json`. Generate this model by running the training section of `model.ipynb`, or provide the model file at that path before running the forecast section.

## Setup

For notebook-based feature preparation and model training, use a Python environment with Jupyter and the notebook dependencies installed. Dashboard/API dependencies are listed separately in `requirements-app.txt`.

```bash
python -m pip install jupyter pandas numpy matplotlib seaborn pyarrow scikit-learn xgboost lightgbm "dask[distributed]" shap statsmodels
python -m pip install -r requirements-app.txt
python -m jupyter notebook
```

Open `EDA.ipynb` first, then `model.ipynb`. Run the notebooks with the project root as the working directory so their relative data and model paths resolve correctly. Feature preparation and XGBoost training are separate from launching the frontend and may require substantial time, memory, and disk space. Do not select **Run All** simply to view existing forecasts.

## Model comparison and selection

The validation comparison below reports MAE for both normal validation (features are built from the observed history) and recursive validation (each prediction is fed back into the history for the next forecast day). Lower MAE is better.

![Bar chart comparing normal and recursive validation MAE for LightGBM, Two-Stage, and XGBoost](assets/model-comparison.png)

| Model | Normal validation MAE | Recursive validation MAE |
| --- | ---: | ---: |
| LightGBM | 0.9741 | 1.1669 |
| Two-Stage (sale occurrence + sale size) | **0.9691** | 1.1453 |
| XGBoost | 0.9706 | **1.1113** |

XGBoost was selected for the final forecast because recursive validation is the closer match to the intended multi-day use: future sales are unknown, so the model must use its own earlier predictions to build subsequent lag and rolling features. XGBoost has the lowest recursive MAE in this comparison (1.1113), about 3.0% lower than Two-Stage (1.1453) and 4.8% lower than LightGBM (1.1669).

This is a metric-based trade-off, not a claim that XGBoost is best in every setting. Two-Stage has the best normal-validation MAE (0.9691), only 0.0015 below XGBoost (0.9706), but its recursive MAE is higher. The Two-Stage approach also requires a sale-occurrence classifier and a conditional sale-size regressor, while the selected XGBoost pipeline uses one regressor for the recursive forecast. Given the substantially better recursive result and simpler final prediction path, XGBoost is the choice for this project.

## Evaluation note

The saved forecast in `data/model_ready/final_forecast_d1914_d1941.parquet` covers `d_1914`–`d_1941` and contains predictions and actual sales for 30,490 item/store series. Recalculated from this file, its metrics are MAE `1.0638`, RMSE `2.2942`, WAPE `73.73%`, and mean bias `-0.1908` (prediction minus actual). These are holdout-period results, not a forecast of dates beyond the available actuals.

On the same holdout, a fixed baseline equal to each series' mean sales over `d_1886`–`d_1913` produced MAE `1.0657`, RMSE `2.2430`, WAPE `73.86%`, and mean bias `-0.0564`.

| Method | MAE | RMSE | WAPE | Mean bias |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 1.0638 | 2.2942 | 73.7308% | -0.1908 |
| Fixed 28-day mean baseline | 1.0657 | 2.2430 | 73.8643% | -0.0564 |

The methods are close on this one holdout: XGBoost has slightly lower MAE and WAPE, while the baseline has lower RMSE and bias closer to zero. This single period does not establish that either method will perform better across other periods. The model comparison above uses a separate validation result and should be interpreted separately.

To recalculate these metrics, run only a notebook cell that reads the saved forecast Parquet and computes the metrics; do not use **Run All** unless you intend to rerun feature preparation and model training. The saved forecast already contains the values needed for evaluation.

## Rolling evaluation and future forecast

The optional `extended_workflows.py` script runs these workflows independently of the notebook:

```bash
python extended_workflows.py backtest
python extended_workflows.py future
```

The rolling backtest evaluates three recursive 28-day periods: `d_1858`–`d_1885`, `d_1886`–`d_1913`, and `d_1914`–`d_1941`. It trains models through each period's cutoff (reusing the saved model trained through `d_1913` for the last fold when its feature count and round count match) and compares them with a fixed mean of the preceding 28 actual days. Results are written to `data/model_ready/rolling_backtest_metrics.csv`. Each fold's XGBoost training uses the existing 150-round, external-memory configuration and may take substantial time.

The future workflow trains a separate XGBoost model using data through `d_1941`, leaving `models/final_xgboost.json` unchanged. It recursively predicts `d_1942`–`d_1969` from the available calendar and sell-price data and saves the model to `models/final_xgboost_d1941.json` and predictions to `data/model_ready/future_forecast_d1942_d1969.parquet`. These are M5 competition dates; actual sales for this forecast period are not included in the evaluation sales file, so these predictions cannot be scored here.

Run both workflows with `python extended_workflows.py all`. Training is computationally intensive; it does not rerun EDA or the full notebook.

### Completed rolling backtest

The three completed folds produced these results:

| Validation period | Method | MAE | RMSE | WAPE | Mean bias |
| --- | --- | ---: | ---: | ---: | ---: |
| `d_1858`–`d_1885` | XGBoost | 1.0401 | 2.3532 | 75.5032% | -0.1487 |
| `d_1858`–`d_1885` | 28-day mean baseline | 1.0458 | 2.3073 | 75.9124% | -0.0005 |
| `d_1886`–`d_1913` | XGBoost | 1.0383 | 2.2515 | 74.8870% | -0.1415 |
| `d_1886`–`d_1913` | 28-day mean baseline | 1.0411 | 2.2186 | 75.0943% | -0.0089 |
| `d_1914`–`d_1941` | XGBoost | 1.0638 | 2.2942 | 73.7308% | -0.1908 |
| `d_1914`–`d_1941` | 28-day mean baseline | 1.0657 | 2.2430 | 73.8643% | -0.0564 |

XGBoost has slightly lower MAE and WAPE in all three folds, while the baseline has lower RMSE and bias closer to zero in all three. The MAE and WAPE differences are small, so this evidence supports XGBoost as a reasonable candidate but not as a decisive improvement over the simple baseline.

The generated future forecast contains 853,720 predictions: 30,490 series for each day from `d_1942` through `d_1969`. Predictions are non-negative and complete in the saved Parquet file. Since actual sales for those dates are unavailable in the supplied data, this forecast is not an accuracy evaluation.

## Interactive dashboard

The Streamlit dashboard uses plain-language date ranges (for example, May 23–June 19, 2016) instead of dataset day codes. It lets users explore estimates by product type, store, and item; compare estimated and actual sales for a past test period; and understand model accuracy without working with notebooks. The final-period M5 projection is clearly marked as a historical example, not a live forecast.

From the project directory, install the dashboard dependencies and start the app:

```powershell
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

The app reads the saved Parquet and CSV outputs; it does not retrain the model. The M5 final-period forecast is a historical dataset forecast, not current real-world demand guidance.

### Optional API

The API is a separate service and is **not required by the current Streamlit frontend**, which reads saved forecast files directly. To run it, open a second PowerShell terminal in the project folder:

```powershell
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health` for service status or `http://127.0.0.1:8000/docs` to try the endpoints. Stop the API with **Ctrl+C**. Running the API does not train a model either.

### How to verify the app and saved model outputs

1. Start Streamlit using the Quick start commands above.
2. In **Explore sales estimates**, choose a saved forecast and verify the date range, chart, and product/store filters.
3. In **Past accuracy**, verify the historical holdout and rolling evaluation values.
4. Try **Replenishment simulator** with sample inventory numbers. It is an assumption-based simulation, not an order recommendation.
5. In **Data quality**, verify the report is loaded. To regenerate it from the raw M5 files, run `python .\data_quality.py`.
6. For API checks, start Uvicorn separately and open `/docs`; `/health` confirms the API process is responsive.

The saved files and the dashboard are connected by file paths, not by live model inference. To check a newly trained model, generate a new forecast with that model, verify its dates and series in the output, and point `app.py` to that forecast file. The current final direct-XGBoost model is a comparison candidate trained through `d_1913`; it is not the selected recursive model and no direct forecast for `d_1942`–`d_1969` has been generated.

## PRD workflow additions

The existing recursive XGBoost model is retained. The saved comparison reports recursive MAE of 1.1113 for XGBoost, 1.1453 for the two-stage candidate, and 1.1669 for LightGBM; two-stage had the best normal-validation MAE. XGBoost was selected for the tested recursive forecast design, not because it is universally better. The comparison does not establish that direct XGBoost beats direct LightGBM. See [MODEL_CARD.md](./MODEL_CARD.md) for intended use and limitations.

The direct 1–28-day XGBoost candidate is separate from the recursive model and does not overwrite its artifacts. Direct training examples use a 56-day observed-sales history at each forecast origin, an explicit horizon feature, target-day calendar/SNAP fields, and origin-known price features. Future realized prices are not used. Model metadata records model, data, and feature-schema hashes.

The direct candidate intentionally omits the notebook's `price_norm`: that value is based on a full-series median, which would expose future price information at earlier forecast origins. Time-since-last-sale is advanced by the requested horizon from the last observed sale.

Run a direct-model evaluation with the training cutoff before the existing holdout:

```powershell
python .\direct_forecasting.py train --training-end-day 1913
python .\direct_forecasting.py forecast --origin-day 1913
python .\direct_forecasting.py evaluate --forecast .\data\model_ready\direct_forecast_d1914_d1941.parquet
```

By default, training creates tens of millions of series/origin/horizon rows and fits 150 boosting rounds. It can take substantial time and disk space. After evaluation, a separate future model can be trained through `d_1941` and forecast from origin `d_1941`:

```powershell
python .\direct_forecasting.py train --training-end-day 1941
python .\direct_forecasting.py forecast --origin-day 1941
```

The `d_1914`–`d_1941` period was already used for recursive analysis and model selection, so direct scores on it are a reused period, not an untouched final test. The matched direct walk-forward run is complete. It retrains one direct model at each fold cutoff and compares it with the existing recursive XGBoost and recent-average results:

| Validation period | Direct model version | Direct MAE | Recursive XGBoost MAE | Recent-average MAE | Direct WAPE | Recursive XGBoost WAPE | Recent-average WAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d_1858`–`d_1885` | `8b39cab39a60` | 1.0666 | 1.0401 | 1.0458 | 77.4288% | 75.5032% | 75.9124% |
| `d_1886`–`d_1913` | `11a813581abf` | 1.0587 | 1.0383 | 1.0411 | 76.3584% | 74.8870% | 75.0943% |
| `d_1914`–`d_1941` | `66f05a514c2e` | 1.0792 | 1.0638 | 1.0657 | 74.8012% | 73.7308% | 73.8643% |

Across these folds, direct XGBoost has higher MAE, RMSE, and WAPE than both recursive XGBoost and the recent-average baseline. Its mean bias is closer to zero than recursive XGBoost, but not closer than the baseline. This experiment therefore does **not** justify replacing the existing recursive forecast with the direct candidate.

The completed run is reproducible from `python .\direct_forecasting.py backtest` and writes per-horizon and aggregate metrics to `data/model_ready/direct_rolling_backtest_metrics.csv`. Every fold used data version `57e4fa85e3b77600a0e9166a23a51fa8e88c6c162fbddf2802d5b8c99bf72689`; the final saved direct model is version `66f05a514c2e`, trained through `d_1913` with 57,199,240 rows, 150 rounds, and feature-schema version `22606a44aec2`. A direct LightGBM comparison and a new untouched later holdout remain outstanding.

Generate the versioned raw-data report used in the dashboard with:

```powershell
python .\data_quality.py
```

The checker verifies sales/calendar schema and continuity, metadata consistency, finite non-negative sales/prices, duplicate price keys across chunk boundaries, and store/item price coverage. It writes versioned JSON reports under `data/quality/`; SQLite is used as a temporary on-disk index for price-key uniqueness.

The local check passed for data version `f2b14f2ea94f1ffead213681493c52b6344ddb53e95096af4f3a5a2bc7158a91`: 30,490 series, 1,941 sales days, 6,841,121 price rows, and no duplicate keys, invalid values, missing sales days, or unmatched store/item price pairs. The generated full report is local under `data/quality/` and can be recreated from the raw CSVs.

Start the API and dashboard in separate terminals:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
python -m streamlit run app.py
```

The API exposes `/health`, `/model`, `/metrics`, `/forecasts`, `/analytics/segments`, `/residuals`, `/feature-importance`, `/data-quality`, and replenishment simulation/history endpoints. Replenishment simulations are persisted in local SQLite. The dashboard adds data-quality and inventory-scenario views and only offers the direct forecast when its saved artifact exists.

Run the focused automated tests with:

```powershell
python -m unittest discover -s tests -v
```

Source code, a direct walk-forward runner, and focused tests are in place. The three-fold direct candidate was trained and evaluated; per-fold forecast and training-example files are local generated data. The small final direct model and its metadata are available under `models/`. A separate final direct model trained through `d_1941` and its `d_1942`–`d_1969` forecast have not been generated. The currently recommended forecast remains recursive XGBoost based on the matched results above.
