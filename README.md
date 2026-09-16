# Store-Level Demand Forecaster

An end-to-end retail demand forecasting and inventory decision-support prototype. It predicts daily observed unit sales for each `store_id` x `item_id` combination for forecast horizons from 1 to 28 days based on the M5 Forecasting Accuracy dataset.

## Architecture and Scope
For detailed architecture flow (HLD/LLD), see [ARCHITECTURE.md](ARCHITECTURE.md). For information regarding the dataset, see [DATASET_OVERVIEW.md](DATASET_OVERVIEW.md).

## Project Structure
```
project_root/
├── data/           # raw, processed, quality_reports, artifacts
├── src/            
│   ├── data/       # ingestion, validation, preprocessing, transformation
│   ├── features/   # calendar, lag, rolling, price, identifiers, aggregates
│   ├── models/     # baselines, lightgbm/quantile/intermittent
│   ├── inventory/  # replenishment, safety_stock
│   └── utils/      # metrics, logger, io, versioning
├── api/            # main, routers, schemas, services
├── dashboard/      # app and pages
├── tests/          # automated test suite (pytest)
├── requirements.txt
├── Dockerfile
└── model_card.md
```

## Quick Start
1. Place raw dataset files in `data/raw/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run FastAPI server: `uvicorn api.main:app --reload`
4. Run Streamlit dashboard: `streamlit run dashboard/app.py`
