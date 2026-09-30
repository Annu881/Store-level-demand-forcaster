import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

def generate_synthetic_m5_data(num_items=5, num_stores=3, num_days=100):
    """Generates a synthetic M5-like long format dataset."""
    print("Generating synthetic dataset...")
    dates = pd.date_range(start='2020-01-01', periods=num_days, freq='D')
    items = [f'ITEM_{i}' for i in range(num_items)]
    stores = [f'STORE_{j}' for j in range(num_stores)]
    
    rows = []
    for item in items:
        for store in stores:
            base_sales = np.random.poisson(lam=np.random.randint(2, 20))
            for i, date in enumerate(dates):
                # Add some noise and weekly seasonality
                seasonality = np.sin(i * (2 * np.pi / 7)) * 2
                sales = max(0, int(base_sales + seasonality + np.random.normal(0, 2)))
                price = round(np.random.uniform(1.0, 15.0), 2)
                
                rows.append({
                    'id': f'{item}_{store}',
                    'item_id': item,
                    'store_id': store,
                    'date': date,
                    'sell_price': price,
                    'sales': sales
                })
                
    df = pd.DataFrame(rows)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    print(f"Generated {len(df)} rows of synthetic data.")
    return df

def engineer_features(df):
    """Creates basic lag and rolling features."""
    print("Engineering features (lags and rolling means)...")
    df = df.sort_values(by=['id', 'date']).reset_index(drop=True)
    
    # Lag features (7 days and 28 days)
    df['lag_7'] = df.groupby('id')['sales'].shift(7)
    df['lag_28'] = df.groupby('id')['sales'].shift(28)
    
    # Rolling features (shifted by 1 to avoid leakage)
    df['rolling_mean_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(7).mean())
    df['rolling_mean_28'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(28).mean())
    
    # Drop NaNs that come from lagging
    df = df.dropna().reset_index(drop=True)
    
    # Categorical encoding
    df['item_id'] = df['item_id'].astype('category')
    df['store_id'] = df['store_id'].astype('category')
    
    return df

def train_demo_model():
    df = generate_synthetic_m5_data(num_items=10, num_stores=3, num_days=150)
    df = engineer_features(df)
    
    # Split data: last 28 days for validation
    max_date = df['date'].max()
    val_start_date = max_date - pd.Timedelta(days=28)
    
    train_df = df[df['date'] < val_start_date]
    val_df = df[df['date'] >= val_start_date]
    
    features = ['sell_price', 'day_of_week', 'is_weekend', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_mean_28', 'item_id', 'store_id']
    target = 'sales'
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    
    # Train LightGBM Model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'seed': 42,
        'verbose': -1
    }
    
    print("Training LightGBM model...")
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=100,
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    # Predictions and Evaluation
    preds = model.predict(X_val)
    # Target cannot be negative
    preds = np.clip(preds, 0, None)
    
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    print("-" * 30)
    print("Evaluation Results on last 28 days:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-" * 30)
    
    # Save the model
    os.makedirs('data/artifacts', exist_ok=True)
    model_path = 'data/artifacts/demo_lgbm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_demo_model()
