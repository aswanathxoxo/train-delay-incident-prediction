import joblib
import uvicorn
from fastapi import FastAPI
import pandas as pd

# Load Saved Pipeline (best_xgb)
model_path = "model_artifacts/tuned_xgboost_v1.0/model.pkl"
model = joblib.load(model_path)

app = FastAPI(title="TNR Incident Prediction API")

# Required Feature Order (37 RAW FEATURES)
FEATURES = [
    "origin","dest","location","weather_main","month","day", "location_part_of_day","origin_part_of_day","dest_part_of_day","train_route_id","dwell_time","on_train_bookings","on_train_forecast","off_train_bookings", "off_train_forecast", "temp", "feels_like", "temp_min","temp_max", "pressure","humidity","wind_speed","rain_1h","snow_1h","clouds_all","year","location_hour","train_service","train_amplitude_record",

    "peak_hour_flag",
    "bad_weather_flag",
    "overcrowded_flag",
    "booking_forecast_error",
    "offboarding_forecast_error",
    "temp_spread",
    "humidity_wind_interaction",
    "rain_or_snow"
]

# -------------------------------
# Root Endpoint
# -------------------------------
@app.get("/")
def home():
    return {"message": "TNR Incident Prediction API is running!"}

# Prediction Endpoint
@app.post("/predict")
def predict(data: dict):

    for f in FEATURES:
        if f not in data:
            return {"error": f"Missing feature: {f}"}

    # Convert input JSON to DataFrame
    df = pd.DataFrame([data], columns=FEATURES)

    # Compute probability of incident
    prob = float(model.predict_proba(df)[0][1])

    # Custom deployment threshold
    THRESHOLD = 0.65   # Adjust 0.60–0.70 based on behaviour

    prediction = 1 if prob >= THRESHOLD else 0

    return {
        "prediction": prediction,
        "probability_of_incident": prob,
        
    }


# Run the API
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=True)
