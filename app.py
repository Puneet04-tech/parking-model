from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load ambient features list for Model 3
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

# Model 1 features (42 features)
model_m1_features = [
    'latitude', 'longitude', 'vehicle_type', 'description', 'center_code',
    'police_station', 'junction_name', 'violation_wrong_parking', 'violation_no_parking',
    'violation_parking_in_a_main_road', 'violation_defective_number_plate',
    'violation_parking_on_footpath', 'violation_parking_near_bustop_school_hospital_etc',
    'violation_double_parking', 'violation_parking_near_road_crossing',
    'violation_refuse_to_go_for_hire', 'violation_parking_near_traffic_light_or_zebra_cross',
    'violation_parking_opposite_to_another_parked_vehicle',
    'violation_using_black_film_other_materials', 'violation_parking_other_than_bus_stop',
    'violation_demanding_excess_fare', 'violation_without_side_mirror', 'violation_count',
    'hour', 'dayofweek', 'month', 'minute', 'is_weekend', 'hour_sin', 'hour_cos',
    'pincode', 'loc_has_metro', 'loc_has_cross', 'loc_has_road', 'loc_has_near',
    'loc_has_opp', 'vehicle_type_freq', 'police_station_freq', 'junction_name_freq',
    'pincode_freq', 'user_target_enc', 'device_target_enc'
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

# Define request schema for validation (Model 1 - 42 features)
class ValidationReport(BaseModel):
    latitude: float
    longitude: float
    vehicle_type: str
    description: float = 0.0
    center_code: float
    police_station: str
    junction_name: str
    violation_wrong_parking: int = 0
    violation_no_parking: int = 0
    violation_parking_in_a_main_road: int = 0
    violation_defective_number_plate: int = 0
    violation_parking_on_footpath: int = 0
    violation_parking_near_bustop_school_hospital_etc: int = 0
    violation_double_parking: int = 0
    violation_parking_near_road_crossing: int = 0
    violation_refuse_to_go_for_hire: int = 0
    violation_parking_near_traffic_light_or_zebra_cross: int = 0
    violation_parking_opposite_to_another_parked_vehicle: int = 0
    violation_using_black_film_other_materials: int = 0
    violation_parking_other_than_bus_stop: int = 0
    violation_demanding_excess_fare: int = 0
    violation_without_side_mirror: int = 0
    violation_count: int = 1
    hour: int
    dayofweek: int
    month: int
    minute: int = 0
    is_weekend: int
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    pincode: str
    loc_has_metro: int = 0
    loc_has_cross: int = 0
    loc_has_road: int = 0
    loc_has_near: int = 0
    loc_has_opp: int = 0
    vehicle_type_freq: float
    police_station_freq: float
    junction_name_freq: float
    pincode_freq: float
    user_target_enc: float = 0.70
    device_target_enc: float = 0.70

# Define request schema for priority prediction (Model 3 - ambient features)
class PriorityReport(BaseModel):
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
def predict_priority(report: PriorityReport):
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
def validate_report(report: ValidationReport):
    """
    Validate whether a parking violation report is legitimate or should be rejected.

    Uses spatial-temporal features and reporter reliability to detect false reports.
    """
    if model_m1 is None:
        raise HTTPException(status_code=503, detail="Validation model not loaded")

    try:
        # Create input DataFrame with all 42 features
        input_data = pd.DataFrame([{
            'latitude': report.latitude,
            'longitude': report.longitude,
            'vehicle_type': report.vehicle_type,
            'description': report.description,
            'center_code': report.center_code,
            'police_station': report.police_station,
            'junction_name': report.junction_name,
            'violation_wrong_parking': report.violation_wrong_parking,
            'violation_no_parking': report.violation_no_parking,
            'violation_parking_in_a_main_road': report.violation_parking_in_a_main_road,
            'violation_defective_number_plate': report.violation_defective_number_plate,
            'violation_parking_on_footpath': report.violation_parking_on_footpath,
            'violation_parking_near_bustop_school_hospital_etc': report.violation_parking_near_bustop_school_hospital_etc,
            'violation_double_parking': report.violation_double_parking,
            'violation_parking_near_road_crossing': report.violation_parking_near_road_crossing,
            'violation_refuse_to_go_for_hire': report.violation_refuse_to_go_for_hire,
            'violation_parking_near_traffic_light_or_zebra_cross': report.violation_parking_near_traffic_light_or_zebra_cross,
            'violation_parking_opposite_to_another_parked_vehicle': report.violation_parking_opposite_to_another_parked_vehicle,
            'violation_using_black_film_other_materials': report.violation_using_black_film_other_materials,
            'violation_parking_other_than_bus_stop': report.violation_parking_other_than_bus_stop,
            'violation_demanding_excess_fare': report.violation_demanding_excess_fare,
            'violation_without_side_mirror': report.violation_without_side_mirror,
            'violation_count': report.violation_count,
            'hour': report.hour,
            'dayofweek': report.dayofweek,
            'month': report.month,
            'minute': report.minute,
            'is_weekend': report.is_weekend,
            'hour_sin': report.hour_sin,
            'hour_cos': report.hour_cos,
            'pincode': report.pincode,
            'loc_has_metro': report.loc_has_metro,
            'loc_has_cross': report.loc_has_cross,
            'loc_has_road': report.loc_has_road,
            'loc_has_near': report.loc_has_near,
            'loc_has_opp': report.loc_has_opp,
            'vehicle_type_freq': report.vehicle_type_freq,
            'police_station_freq': report.police_station_freq,
            'junction_name_freq': report.junction_name_freq,
            'pincode_freq': report.pincode_freq,
            'user_target_enc': report.user_target_enc,
            'device_target_enc': report.device_target_enc
        }])

        # Convert categorical columns to category type
        cat_cols = ['vehicle_type', 'police_station', 'junction_name', 'pincode']
        for col in cat_cols:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(str).astype('category')

        # Ensure all expected features are present
        for feature in model_m1_features:
            if feature not in input_data.columns:
                input_data[feature] = 0

        # Reorder columns to match training data
        input_data = input_data[model_m1_features]

        # Make prediction
        prediction = int(model_m1.predict(input_data)[0])
        prediction_proba = model_m1.predict_proba(input_data)[0]
        confidence = float(prediction_proba[prediction])

        # Map prediction to status
        status = "APPROVED" if prediction == 1 else "REJECTED"

        return {
            "success": True,
            "validation_status": status,
            "confidence": confidence,
            "prediction_code": prediction,
            "model_version": "1.0.0"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
