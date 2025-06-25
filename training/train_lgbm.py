import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocessing.data_loader import load_all_data
from preprocessing.feature_engineering import prepare_features
import numpy as np
import os
import warnings
import joblib
import sys
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


data = load_all_data()
train_df = data["train"]
oil_df = data["oil"]
holidays_df = data["holidays"]
transactions_df = data["transactions"]
stores_df = data["stores"]


train_df["id"] = train_df["store_nbr"].astype(str) + "_" + train_df["family"].astype(str)


train_df = prepare_features(train_df, oil_df, holidays_df, transactions_df, stores_df)
train_df.dropna(inplace=True)


features = [
    'store_nbr', 'family', 'onpromotion', 'year', 'month', 'day',
    'dayofweek', 'weekofyear', 'quarter', 'city', 'state', 'type',
    'cluster', 'dcoilwtico', 'transactions', 'is_holiday',
    'lag_7', 'lag_14', 'rolling_mean_7'
]
target = 'sales'


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['store_nbr', 'family', 'city', 'state', 'type']:
    train_df[col] = le.fit_transform(train_df[col])


X = train_df[features]
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)


val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"LightGBM RMSE: {rmse:.2f}")

# Save model and related components
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/lgbm_model.pkl")
joblib.dump(features, "models/lgbm_features.pkl")
encoders = {col: LabelEncoder().fit(train_df[col]) for col in ['store_nbr', 'family', 'city', 'state', 'type']}
joblib.dump(encoders, "models/lgbm_encoders.pkl")
print("LightGBM model, features and encoders saved successfully.")