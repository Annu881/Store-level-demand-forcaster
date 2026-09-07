# Store-Level Demand Forecaster Architecture

This document contains the High-Level Design (HLD) and Low-Level Design (LLD) diagrams for the Store-Level Demand Forecaster system.

## High-Level Design (HLD)

The HLD illustrates the end-to-end workflow covering offline model training and the online presentation/inference layer.

```mermaid
flowchart TB
    subgraph Data_Sources ["Data Sources"]
        M5_CSV[(M5 CSV Files\nSales, Calendar, Prices)]
        External_POS[(Future POS / ERP\nFor Extension)]
    end

    subgraph Offline_Training_Pipeline ["Offline Training Pipeline"]
        DI[Data Ingestion & Validation]
        Transform[Wide-to-Long Transformation]
        FE[Feature Engineering\nCalendar, Lags, Price]
        Split[Time-based Split]
        Train[Global LightGBM Training & Baselines]
        Eval[Walk-Forward Evaluation\n& Final Holdout]
        Registry[(Model Artifacts & Versions)]
        
        M5_CSV --> DI
        DI --> Transform
        Transform --> FE
        FE --> Split
        Split --> Train
        Train --> Eval
        Eval --> Registry
    end

    subgraph Serving_API_Layer ["Serving & API Layer (Online)"]
        API[FastAPI Endpoints]
        Inference[Forecast Service & Model Inference]
        Sim[Replenishment Simulator]
        DB[(Persistence\nSQLite/Parquet)]
        
        API <--> Inference
        API <--> Sim
        Inference --> Registry
        Inference --> DB
        Sim --> DB
    end

    subgraph User_Interface ["User Interface"]
        Streamlit[Streamlit Dashboard\nForecast Explorer, Simulator, Analytics]
    end

    Streamlit <--> API
```

## Low-Level Design (LLD)

### 1. Data & Feature Pipeline

```mermaid
flowchart LR
    Raw[Raw M5 CSVs] --> Validator{Schema & Quality Checks}
    Validator -- Passed --> LongFormat[Wide-to-Long DataFrame]
    Validator -- Failed --> Reject(Log Error / Report)

    LongFormat --> CalMerge[Merge Calendar Data]
    CalMerge --> PriceMerge[Merge Sell Prices]
    PriceMerge --> FeatureEng
    
    subgraph Feature_Engineering ["Feature Engineering"]
        direction TB
        CalFeat(Calendar: known events, SNAP)
        HistFeat(Lags 1/7/14/28, Rolling means)
        PriceFeat(Last known price, Price lag)
        IdFeat(Store, Item, Dept identifiers)
    end
    FeatureEng --> CalFeat & HistFeat & PriceFeat & IdFeat
```

### 2. Inference & Replenishment Flow

```mermaid
sequenceDiagram
    participant User as Streamlit User
    participant Dash as Streamlit Dashboard
    participant API as FastAPI
    participant Model as Forecast Service
    participant Sim as Inventory Simulator
    participant DB as SQLite / Parquet

    User->>Dash: Inputs Store, Item, Horizon
    Dash->>API: POST /api/v1/forecasts
    API->>Model: Build origin-available features
    Model->>Model: Run LightGBM Inference
    Model->>Model: Clip at Zero: max(0, pred)
    Model->>DB: Persist Run & Forecasts
    API-->>Dash: Return Forecasts json
    Dash-->>User: Display Forecasts limits

    User->>Dash: Inputs Inventory Assumptions (On-hand, LT, etc.)
    Dash->>API: POST /api/v1/replenishment/simulate
    API->>Sim: Calculate Target Stock & Safety Stock
    Sim->>Sim: Apply MOQ & Pack Size adjustments
    Sim->>DB: Persist Scenario & Recommendations
    API-->>Dash: Return Simulated Recommendation
    Dash-->>User: Show Alerts & Scenario (with Disclaimer)
```

## Functional Components Overview

- **Storage**: M5 dataset input and SQLite/Parquet for saving forecasts and scenarios.
- **Model Training**: Extracts `sales_train_validation.csv`, validates formatting. Performs walk-forward multi-horizon forecasting `f(X_t, h) -> y_(t+h)`.
- **FastAPI Endpoints**: 
  - `GET /api/v1/health`, `GET /api/v1/model/info`, `GET /api/v1/model/metrics`
  - `POST /api/v1/forecasts` to generate a forecast request.
  - `POST /api/v1/replenishment/simulate` for custom user-parameters to simulate expected requirements.
- **Dashboard (Streamlit)**: Forecast explorer, model metrics view, configuration/scenario tuning.
