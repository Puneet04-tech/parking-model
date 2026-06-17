from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json

# Initialize FastAPI app
app = FastAPI(
    title="Parking Congestion Intelligence API",
    description="Live inference engine for report validation and congestion priority dispatch.",
    version="1.0.0"
)

# Load ambient features list
try:
    with open('ambient_features.json', 'r') as f:
        ambient_features = json.load(f)
    print("✅ Ambient features loaded successfully")
except Exception as e:
    print(f"⚠️ Warning loading ambient features: {e}")
    ambient_features = [
        'latitude', 'longitude',
        'hour_sin', 'hour_cos', 'dayofweek', 'is_weekend', 'month',
        'police_station_freq', 'pincode_freq', 'junction_name_freq',
        'user_target_enc', 'device_target_enc'
    ]

# Load production models
try:
    model_m1 = joblib.load('model_report_validator.joblib')
    model_m3 = joblib.load('model_congestion_prioritizer.joblib')
    print("📦 Production models successfully loaded into memory!")
except Exception as e:
    print(f"🚨 Critical Error loading models: {str(e)}")
    model_m1 = None
    model_m3 = None

# Define request schema
class ParkingReport(BaseModel):
    latitude: float
    longitude: float
    hour: int
    dayofweek: int
    is_weekend: int
    month: int
    police_station_freq: float
    pincode_freq: float
    junction_name_freq: float = 0.0
    user_target_enc: float = 0.70
    device_target_enc: float = 0.70

# Priority mapping
PRIORITY_MAPPING = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Parking Intelligence Engine Running",
        "version": "1.0.0",
        "models_loaded": model_m1 is not None and model_m3 is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_validator_loaded": model_m1 is not None,
        "model_prioritizer_loaded": model_m3 is not None
    }

@app.post("/predict_priority")
def predict_priority(report: ParkingReport):
    """
    Predict enforcement priority for a parking violation report.
    
    Uses ambient features (location, time, area density) to predict
    congestion impact without direct access to vehicle type or violation details.
    """
    if model_m3 is None:
        raise HTTPException(status_code=503, detail="Priority model not loaded")
    
    try:
        # Create input DataFrame matching model's expected format
        input_data = pd.DataFrame([{
            'latitude': report.latitude,
            'longitude': report.longitude,
            'hour_sin': np.sin(2 * np.pi * report.hour / 24.0),
            'hour_cos': np.cos(2 * np.pi * report.hour / 24.0),
            'dayofweek': report.dayofweek,
            'is_weekend': report.is_weekend,
            'month': report.month,
            'police_station_freq': report.police_station_freq,
            'pincode_freq': report.pincode_freq,
            'junction_name_freq': report.junction_name_freq,
            'user_target_enc': report.user_target_enc,
            'device_target_enc': report.device_target_enc
        }])
        
        # Ensure all expected features are present
        for feature in ambient_features:
            if feature not in input_data.columns:
                input_data[feature] = 0.0
        
        # Reorder columns to match training data
        input_data = input_data[ambient_features]
        
        # Make prediction
        prediction_id = int(model_m3.predict(input_data)[0])
        priority_text = PRIORITY_MAPPING.get(prediction_id, "UNKNOWN")
        
        # Get prediction confidence
        prediction_proba = model_m3.predict_proba(input_data)[0]
        confidence = float(prediction_proba[prediction_id])
        
        # Determine enforcement action
        enforcement_action = "DISPATCH_PATROL" if priority_text in ["HIGH"] else "MONITOR_DASHBOARD"
        
        return {
            "success": True,
            "prediction_code": prediction_id,
            "recommended_dispatch_priority": priority_text,
            "confidence": confidence,
            "enforcement_action": enforcement_action,
            "model_version": "1.0.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.post("/validate_report")
def validate_report(report: ParkingReport):
    """
    Validate whether a parking violation report is legitimate or should be rejected.
    
    Uses spatial-temporal features and reporter reliability to detect false reports.
    """
    if model_m1 is None:
        raise HTTPException(status_code=503, detail="Validation model not loaded")
    
    try:
        # For validation, we would need additional features from the original dataset
        # This is a simplified version - in production, you'd need all Model 1 features
        
        input_data = pd.DataFrame([{
            'latitude': report.latitude,
            'longitude': report.longitude,
            'hour': report.hour,
            'dayofweek': report.dayofweek,
            'is_weekend': report.is_weekend,
            'month': report.month,
            'police_station_freq': report.police_station_freq,
            'pincode_freq': report.pincode_freq,
            'junction_name_freq': report.junction_name_freq,
            'user_target_enc': report.user_target_enc,
            'device_target_enc': report.device_target_enc
        }])
        
        # Note: This is a placeholder - actual validation requires all 42 features from Model 1
        # In production, ensure your API receives all required features
        
        return {
            "success": True,
            "message": "Validation requires full feature set (42 features)",
            "note": "This endpoint is a placeholder - implement with full Model 1 features"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
