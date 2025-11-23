import joblib
import pandas as pd
import numpy as np

# Path to the trained model
MODEL_PATH = 'models/inventory_model.pkl'

def load_model():
    """Loads the trained model from the pkl file."""
    try:
        model = joblib.load(MODEL_PATH)
        print("AI Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Did you run the training notebook?")
        return None

def make_predictions(model, df_uploaded_excel):
    """
    Receives DataFrame from the uploaded Excel file and returns prediction results.
    
    In a real project, df_uploaded_excel would be used to calculate the 
    current stock, find the last usage date, and determine future prediction features.
    """
    if model is None:
        return {"CATH-SKU-ERR": {"status": "error", "message": "Model not loaded."}}
    
    # --- 1. Simulation of creating future prediction input ---
    # We assume the last day of historical data was day index 100.
    latest_day_index = 100 
    
    # Predict for the next 7 days
    future_days = pd.DataFrame({'day_index': range(latest_day_index + 1, latest_day_index + 8)})
    
    # --- 2. Predict Future Usage ---
    future_usage_prediction = model.predict(future_days[['day_index']])
    
    # Calculate recommended quantity (Usage prediction * Safety Factor 1.5)
    recommended_units = int(future_usage_prediction.sum() * 1.5)
    estimated_cost = int(recommended_units * 150) # Assuming 150/unit
    
    # --- 3. Create Action Plan for Mock SKU ---
    
    predictions = {
        "CATH-101A": {
            "recommended_quantity": recommended_units,
            "estimated_cost": estimated_cost, 
            "reason": f"Forecasted usage for the next 7 days is high. Recommend reordering {recommended_units} units.",
            "priority": "HIGH"
        },
        "GUIDE-007X": {
            "recommended_quantity": 0,
            "estimated_cost": 0,
            "reason": "Stock level is sufficient based on current trends. Monitoring.",
            "priority": "LOW"
        }
    }

    return predictions