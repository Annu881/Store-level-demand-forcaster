"""
baselines.py
Computes 3 baseline forecasts on real M5 sampled data and compares them against LightGBM.
Baselines:
  1. Naive        → predict last known sales value
  2. Moving Avg 7 → mean of previous 7 days
  3. Seasonal Naive → same weekday 7 days ago
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW_DIR   = '../data/raw'
N_STORES  = 3
N_ITEMS   = 50
MODEL_PKL = 'data/artifacts/real_m5_lgbm_model.pkl'

# ── 1. Load real M5 data (same slice as training) ─────────────────────────────
def load_data():
    print("Loading real M5 data...")
    sales  = pd.read_csv(f'{RAW_DIR}/sales_train_validation.csv')
    cal    = pd.read_csv(f'{RAW_DIR}/calendar.csv')
    prices = pd.read_csv(f'{RAW_DIR}/sell_prices.csv')

    stores = sales['store_id'].unique()[:N_STORES]
    items  = sales['item_id'].unique()[:N_ITEMS]
    sales  = sales[sales['store_id'].isin(stores) & sales['item_id'].isin(items)]

    id_cols  = ['id','item_id','dept_id','cat_id','store_id','state_id']
    day_cols = [c for c in sales.columns if c.startswith('d_')]
    df = pd.melt(sales, id_vars=id_cols, value_vars=day_cols, var_name='d', value_name='sales')

    cal_cols = ['d','date','wm_yr_wk','wday','month','year','snap_CA','snap_TX','snap_WI']
    df = df.merge(cal[cal_cols], on='d', how='left')
    df = df.merge(prices, on=['store_id','item_id','wm_yr_wk'], how='left')
    df['date']       = pd.to_datetime(df['date'])
    df['sales']      = df['sales'].astype(float)
    df['sell_price'] = df['sell_price'].astype(float)
    for c in ['item_id','dept_id','cat_id','store_id','state_id']:
        df[c] = df[c].astype('category')
    df = df.sort_values(['id','date']).reset_index(drop=True)
    return df

# ── 2. Build the same lag features (needed for LightGBM re-prediction) ────────
def add_features(df):
    grp = df.groupby('id')['sales']
    df['lag_1']          = grp.shift(1)
    df['lag_7']          = grp.shift(7)
    df['lag_28']         = grp.shift(28)
    df['rolling_mean_7'] = grp.transform(lambda x: x.shift(1).rolling(7).mean())
    df['rolling_mean_28']= grp.transform(lambda x: x.shift(1).rolling(28).mean())
    # seasonal naive: same weekday 7 days ago
    df['seasonal_naive'] = grp.shift(7)
    return df

# ── 3. Evaluation helper ──────────────────────────────────────────────────────
def evaluate(name, actual, predicted):
    predicted = np.clip(predicted, 0, None)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"  {name:<22} | MAE: {mae:.4f}  | RMSE: {rmse:.4f}")
    return {'model': name, 'MAE': round(mae,4), 'RMSE': round(rmse,4)}

# ── 4. Main comparison ────────────────────────────────────────────────────────
def run():
    df = load_data()
    df = add_features(df)
    df.dropna(inplace=True)

    # Validation = last 28 days
    max_date  = df['date'].max()
    val_start = max_date - pd.Timedelta(days=28)
    val = df[df['date'] >= val_start].copy()

    actual = val['sales'].values
    results = []

    print("\n" + "="*55)
    print("  Baseline vs LightGBM Comparison (last 28-day holdout)")
    print("="*55)

    # Baseline 1: Naive (yesterday's value)
    results.append(evaluate("Naive (lag-1)",      actual, val['lag_1'].values))

    # Baseline 2: 7-day Moving Average
    results.append(evaluate("Moving Avg (7-day)", actual, val['rolling_mean_7'].values))

    # Baseline 3: Seasonal Naive (same weekday last week)
    results.append(evaluate("Seasonal Naive",     actual, val['seasonal_naive'].values))

    # LightGBM prediction using saved model
    FEATURES = ['item_id','dept_id','cat_id','store_id','state_id',
                'wday','month','year','snap_CA','snap_TX','snap_WI',
                'sell_price','lag_7','lag_28','rolling_mean_7','rolling_mean_28']
    try:
        with open(MODEL_PKL, 'rb') as f:
            model = pickle.load(f)
        lgb_preds = model.predict(val[FEATURES])
        results.append(evaluate("LightGBM (real M5)", actual, lgb_preds))
    except FileNotFoundError:
        print("  LightGBM model not found. Run train_real_sampled.py first.")

    print("="*55)

    # Summary table
    print("\n📊 Summary Table:")
    print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*45}")
    for r in results:
        marker = " ✅ BEST" if r['MAE'] == min(x['MAE'] for x in results) else ""
        print(f"  {r['model']:<25} {r['MAE']:>8.4f} {r['RMSE']:>8.4f}{marker}")
    print()

if __name__ == '__main__':
    run()
