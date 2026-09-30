"""
train_real_sampled.py
Trains LightGBM on REAL M5 data but a safe memory-limited sample.
Instead of loading all 30,490 series, we pick a configurable subset of stores and items.
This is the correct way to train on real data without OOM crashing a standard laptop.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

# ── CONFIG ───────────────────────────────────────────────────────────────────
RAW_DIR      = '../data/raw'   # real Kaggle CSVs live here (outer folder)
N_STORES     = 3               # how many stores to use  (max 10)
N_ITEMS      = 50              # how many items per store (max ~3049)
ARTIFACTS    = 'data/artifacts'
# ─────────────────────────────────────────────────────────────────────────────

def load_sampled_data():
    print("Loading real M5 CSV files (sampled)...")
    sales_wide = pd.read_csv(f'{RAW_DIR}/sales_train_validation.csv')
    calendar   = pd.read_csv(f'{RAW_DIR}/calendar.csv')
    prices     = pd.read_csv(f'{RAW_DIR}/sell_prices.csv')

    # ── pick a safe subset of stores & items ──────────────────────────────
    stores = sales_wide['store_id'].unique()[:N_STORES]
    print(f"Using stores: {list(stores)}")
    sales_wide = sales_wide[sales_wide['store_id'].isin(stores)]

    items = sales_wide['item_id'].unique()[:N_ITEMS]
    print(f"Using {len(items)} items.")
    sales_wide = sales_wide[sales_wide['item_id'].isin(items)]

    # ── Wide → Long ────────────────────────────────────────────────────────
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    day_cols = [c for c in sales_wide.columns if c.startswith('d_')]

    print("Melting to long format...")
    df = pd.melt(sales_wide, id_vars=id_cols, value_vars=day_cols,
                 var_name='d', value_name='sales')

    # ── Merge calendar ─────────────────────────────────────────────────────
    cal_cols = ['d', 'date', 'wm_yr_wk', 'wday', 'month', 'year',
                'snap_CA', 'snap_TX', 'snap_WI']
    df = df.merge(calendar[cal_cols], on='d', how='left')

    # ── Merge prices ───────────────────────────────────────────────────────
    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # ── Memory optimization ────────────────────────────────────────────────
    df['date']       = pd.to_datetime(df['date'])
    df['sales']      = df['sales'].astype(np.float32)
    df['sell_price'] = df['sell_price'].astype(np.float32)
    for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
        df[col] = df[col].astype('category')

    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows from REAL M5 data.")
    return df


def engineer_features(df):
    print("Engineering leakage-safe lag and rolling features...")
    grp = df.groupby('id')['sales']
    df['lag_7']          = grp.shift(7).astype(np.float32)
    df['lag_28']         = grp.shift(28).astype(np.float32)
    df['rolling_mean_7'] = grp.transform(lambda x: x.shift(1).rolling(7).mean()).astype(np.float32)
    df['rolling_mean_28']= grp.transform(lambda x: x.shift(1).rolling(28).mean()).astype(np.float32)
    df.dropna(inplace=True)
    return df


def train():
    df = load_sampled_data()
    df = engineer_features(df)

    # ── Time-based walkforward split ───────────────────────────────────────
    max_date  = df['date'].max()
    val_start = max_date - pd.Timedelta(days=28)
    train_df  = df[df['date'] < val_start]
    val_df    = df[df['date'] >= val_start]

    FEATURES = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                'wday', 'month', 'year',
                'snap_CA', 'snap_TX', 'snap_WI',
                'sell_price', 'lag_7', 'lag_28',
                'rolling_mean_7', 'rolling_mean_28']
    TARGET = 'sales'

    X_tr, y_tr = train_df[FEATURES], train_df[TARGET]
    X_va, y_va = val_df[FEATURES],   val_df[TARGET]
    print(f"Train: {len(X_tr):,} rows | Validation: {len(X_va):,} rows")

    # ── LightGBM training ──────────────────────────────────────────────────
    params = {
        'objective': 'regression',
        'metric':    'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves':    63,
        'seed': 42,
        'verbose': -1
    }

    print("Training Global LightGBM on REAL M5 data...")
    model = lgb.train(
        params,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=50)]
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    preds = np.clip(model.predict(X_va), 0, None)
    mae   = mean_absolute_error(y_va, preds)
    rmse  = np.sqrt(mean_squared_error(y_va, preds))

    print("\n" + "="*40)
    print(" REAL M5 Model Evaluation (last 28-day holdout)")
    print("="*40)
    print(f"  Stores used  : {N_STORES}")
    print(f"  Items used   : {N_ITEMS}")
    print(f"  MAE          : {mae:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print("="*40)

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(ARTIFACTS, exist_ok=True)
    path = f'{ARTIFACTS}/real_m5_lgbm_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")

if __name__ == '__main__':
    train()
