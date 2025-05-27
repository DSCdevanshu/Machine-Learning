import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Load data and model with error handling
try:
    test_data = pd.read_pickle('test_data.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Error: 'test_data.pkl' file not found. Ensure it exists in the correct directory.")
except Exception as e:
    raise Exception(f"Error loading test_data.pkl: {str(e)}")

try:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('xgb_model.json')
except FileNotFoundError:
    raise FileNotFoundError("Error: 'xgb_model.json' file not found. Ensure it exists in the correct directory.")
except Exception as e:
    raise Exception(f"Error loading xgb_model.json: {str(e)}")

features = ['shop_id', 'item_id', 'item_category_id', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'price_trend']

# Validate features in test_data
missing_features = [feat for feat in features if feat not in test_data.columns]
if missing_features:
    raise ValueError(f"Error: Missing features in test_data: {missing_features}")

@app.get("/predict/{shop_id}/{item_id}")
async def predict_sales(shop_id: int, item_id: int):
    try:
        input_data = test_data[(test_data['shop_id'] == shop_id) & (test_data['item_id'] == item_id)]
        if input_data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given shop_id and item_id")
        
        # Replace any infinite values in input_data
        input_data[features] = input_data[features].replace([float('inf'), -float('inf')], 0)
        
        pred = xgb_model.predict(input_data[features])[0]
        pred = max(pred, 0)
        return {"shop_id": shop_id, "item_id": item_id, "predicted_sales": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# Optional: Add a root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the Sales Prediction API"}