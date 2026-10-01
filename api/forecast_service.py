"""
forecast_service.py
Loads the trained model and builds origin-available features for inference.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "artifacts" / "real_m5_lgbm_model.pkl"
RAW_DIR    = Path(__file__).resolve().parents[1].parent / "data" / "raw"

FEATURES = [
    'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
    'wday', 'month', 'year',
    'snap_CA', 'snap_TX', 'snap_WI',
    'sell_price', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_mean_28'
]

# Module-level model cache
_model = None
MODEL_VERSION = "real_m5_lgbm_v1"

def load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model

def is_model_loaded() -> bool:
    try:
        load_model()
        return True
    except Exception:
        return False

def _get_history_for_item(store_id: str, item_id: str) -> pd.DataFrame:
    """Loads the last 56 days of sales for a given store-item pair from M5 CSVs."""
    sales  = pd.read_csv(RAW_DIR / "sales_train_validation.csv")
    cal    = pd.read_csv(RAW_DIR / "calendar.csv")
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv")

    row = sales[(sales['store_id'] == store_id) & (sales['item_id'] == item_id)]
    if row.empty:
        raise ValueError(f"No data found for store={store_id}, item={item_id}")

    id_cols  = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    day_cols = [c for c in sales.columns if c.startswith('d_')]
    df = pd.melt(row, id_vars=id_cols, value_vars=day_cols, var_name='d', value_name='sales')

    cal_cols = ['d','date','wm_yr_wk','wday','month','year','snap_CA','snap_TX','snap_WI']
    df = df.merge(cal[cal_cols], on='d', how='left')
    df = df.merge(prices, on=['store_id','item_id','wm_yr_wk'], how='left')
    df['date']       = pd.to_datetime(df['date'])
    df['sales']      = df['sales'].astype(float)
    df['sell_price'] = df['sell_price'].fillna(method='ffill').astype(float)
    df = df.sort_values('date').reset_index(drop=True)
    return df

def generate_forecasts(store_id: str, item_id: str, horizon_days: int = 28):
    model  = load_model()
    hist   = _get_history_for_item(store_id, item_id)

    # Extract item metadata
    meta = hist.iloc[-1]
    dept_id  = str(meta['dept_id'])
    cat_id   = str(meta['cat_id'])
    state_id = str(meta['state_id'])
    last_price = float(meta['sell_price']) if not pd.isna(meta['sell_price']) else 0.0

    sales_history = hist['sales'].tolist()
    last_date = hist['date'].max()

    forecasts = []
    for h in range(1, horizon_days + 1):
        target_date  = last_date + timedelta(days=h)
        lag_7        = sales_history[-7 + h - 1] if len(sales_history) >= 7 else 0.0
        lag_28       = sales_history[-28 + h - 1] if len(sales_history) >= 28 else 0.0
        roll_7 = float(np.mean(sales_history[max(0, len(sales_history)-7):]))
        roll_28= float(np.mean(sales_history[max(0, len(sales_history)-28):]))

        row = pd.DataFrame([{
            'item_id': item_id, 'dept_id': dept_id, 'cat_id': cat_id,
            'store_id': store_id, 'state_id': state_id,
            'wday': target_date.weekday() + 1,
            'month': target_date.month, 'year': target_date.year,
            'snap_CA': 0, 'snap_TX': 0, 'snap_WI': 0,
            'sell_price': last_price,
            'lag_7': lag_7, 'lag_28': lag_28,
            'rolling_mean_7': roll_7, 'rolling_mean_28': roll_28
        }])

        for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
            row[col] = row[col].astype('category')

        raw_pred      = float(model.predict(row[FEATURES])[0])
        point_forecast = max(0.0, raw_pred)
        sales_history.append(point_forecast)
        forecasts.append({
            'forecast_date': target_date.date(),
            'horizon': h,
            'point_forecast': round(point_forecast, 4)
        })

    return forecasts
