# Store-Level Demand Forecaster

This project explores and forecasts daily retail unit sales for item/store series using the M5 Forecasting dataset. The workflow is implemented in two Jupyter notebooks: exploratory data analysis and feature creation, followed by model experiments, final training, and a recursive 28-day forecast.

## Project files

```text
.
├── EDA.ipynb                 # Dataset exploration, feature engineering, and model-ready data
├── model.ipynb               # Baselines, model experiments, training, and forecasting
├── data/
│   ├── raw/                  # M5 source CSV files and intermediate training features
│   └── model_ready/          # Engineered features, feature configuration, forecast output
└── models/
    ├── final_xgboost.json    # Trained XGBoost model used for forecasting
    └── xgb_cache/            # XGBoost external-memory training cache
```

The data and model files are generated or supplied locally and may not be included in a Git checkout. The source M5 data is available from the [Kaggle M5 Forecasting Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

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

Use a Python environment with Jupyter and the notebook dependencies installed:

```bash
pip install jupyter pandas numpy matplotlib seaborn pyarrow scikit-learn xgboost lightgbm "dask[distributed]" shap statsmodels
jupyter notebook
```

Open `EDA.ipynb` first, then `model.ipynb`. Run the notebooks with the repository root as the working directory so their relative data and model paths resolve correctly. The XGBoost training and feature dataset require substantial disk space and memory.

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

The model notebook records an initial XGBoost result for the 28-day forecast period of MAE `1.0629`, RMSE `2.1628`, and WAPE `73.67%`. It reports a moving-average-28 baseline MAE of `1.0641`. These are notebook-reported results, not a guarantee that a fresh run with different data, dependencies, or model settings will produce identical metrics.
