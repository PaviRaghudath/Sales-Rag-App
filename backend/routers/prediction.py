import joblib
import pandas as pd

# Load saved model
model = joblib.load("models/lgbm_model.pkl")

def predict_sales(input_dict):
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)
