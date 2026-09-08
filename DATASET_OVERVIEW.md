# M5 Forecasting - Accuracy: Dataset Overview

This project utilizes the **M5 Forecasting - Accuracy** dataset, based on the Kaggle competition hosted in 2020 by the Makridakis Open Forecasting Center (MOFC) at the University of Nicosia, in partnership with Walmart. 

The primary objective is to estimate the point forecasts of the daily unit sales of various products sold in the USA by Walmart for the next 28 days.

## Dataset Characteristics

- **Source:** Walmart (World's largest company by revenue)
- **Time Period:** 2020 (Competition Timeline), containing historical daily sales spanning over 5 years.
- **Scope:** Hierarchical sales data covering stores across three US States (California, Texas, and Wisconsin).
- **Features:** 
  - Item level details
  - Department and product categories
  - Store details
  - Explanatory variables: price, promotions, day of the week, and special events.

## Data Files

The dataset is divided into several core CSV files used for training and evaluating the models:

### 1. `sales_train_validation.csv` (Training Data)
- Contains the historical daily unit sales data per product and store.
- **Timeframe:** `[d_1 - d_1913]`
- **Purpose:** Used as the primary training data to build the demand forecasting models.

### 2. `sales_train_evaluation.csv` (Evaluation Data)
- Includes sales for an extended period, containing slightly more historical data.
- **Timeframe:** `[d_1 - d_1941]`
- **Purpose:** Used for model evaluation. These were the labels used for the Public leaderboard during the competition phase.

### 3. `calendar.csv`
- Contains contextual information about the dates on which the products are sold.
- **Details:** Includes dates, weekdays, months, years, and specific information regarding special events and SNAP (Supplemental Nutrition Assistance Program) days.

### 4. `sell_prices.csv`
- Contains information about the historical sell price of the products.
- **Details:** Weekly pricing per store and date for each item combination.

### 5. `sample_submission.csv`
- Provides the correct format expected for forecasting outputs.
- **Format:** Each row contains an `id` (concatenated `item_id` and `store_id`), followed by `F1` to `F28` representing the predictions for the upcoming 28 forecast days.

## Challenge Goal
To use machine learning (such as Global LightGBM) and traditional forecasting methods to predict 28 days of future unit sales (F1-F28). Performance is evaluated using a Custom Metric: **Weighted Root Mean Squared Scaled Error (RMSSE)**.
