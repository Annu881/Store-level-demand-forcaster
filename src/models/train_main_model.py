import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import gc

def load_and_transform_data(raw_dir='../data/raw'):
    print("Loading datasets...")
    try:
        sales = pd.read_csv(f'{raw_dir}/sales_train_validation.csv')
        calendar = pd.read_csv(f'{raw_dir}/calendar.csv')
        prices = pd.read_csv(f'{raw_dir}/sell_prices.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have downloaded the M5 Kaggle dataset and placed 'sales_train_validation.csv', 'calendar.csv', and 'sell_prices.csv' into the 'data/raw' directory.")
        return None

    # Wide to Long transformation
    print("Melting sales data to long format...")
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_long = pd.melt(sales, id_vars=id_vars, var_name='d', value_name='sales')
    
    # Free memory
    del sales
    gc.collect()

    print("Merging calendar and price features...")
    # Merge Calendar
    sales_long = sales_long.merge(calendar[['d', 'date', 'wm_yr_wk', 'wday', 'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI']], on='d', how='left')
    
    # Merge Prices
    sales_long = sales_long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # Reduce memory sizes
    sales_long['date'] = pd.to_datetime(sales_long['date'])
    sales_long['sales'] = sales_long['sales'].astype(np.float32)
    sales_long['sell_price'] = sales_long['sell_price'].astype(np.float32)
    
    for col in id_vars + ['wday', 'month', 'year']:
        sales_long[col] = sales_long[col].astype('category')

    # Sort sequentially
    sales_long = sales_long.sort_values(by=['id', 'date']).reset_index(drop=True)
    return sales_long

def engineer_features(df):
    print("Engineering lag and rolling features...")
    # Lags
    df['lag_7'] = df.groupby('id')['sales'].shift(7).astype(np.float32)
    df['lag_28'] = df.groupby('id')['sales'].shift(28).astype(np.float32)

    # Rolling means (shift 1 to avoid leakage)
    df['rolling_mean_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(7).mean()).astype(np.float32)
    df['rolling_mean_28'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(28).mean()).astype(np.float32)
    
    print("Dropping initial NaN rows due to lagging...")
    df.dropna(inplace=True)
    return df

def train_model():
    df = load_and_transform_data()
    if df is None:
        return

    df = engineer_features(df)
    
    # Train-Var split
    print("Splitting data (last 28 days for validation)...")
    max_date = df['date'].max()
    val_start_date = max_date - pd.Timedelta(days=28)
    
    train_df = df[df['date'] < val_start_date]
    val_df = df[df['date'] >= val_start_date]
    
    features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
        'wday', 'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI',
        'sell_price', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_mean_28'
    ]
    target = 'sales'
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 63,
        'seed': 42,
        'verbose': 1
    }
    
    print("Training Global LightGBM model...")
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Validation Eval
    preds = model.predict(X_val)
    preds = np.clip(preds, 0, None)
    
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    print("-" * 30)
    print("Main Model Evaluation Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-" * 30)
    
    # Save the model
    os.makedirs('../../data/artifacts', exist_ok=True)
    model_path = '../../data/artifacts/main_lgbm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Main model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()
