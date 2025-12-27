import sys
# --- NEW CODE: BLOCK PYSPARK ON WINDOWS ---
# This tricks 'shap' into thinking pyspark is not installed
sys.modules["pyspark"] = None
import pandas as pd
import joblib
import logging
import shap
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")
MODEL_PATH = Path("models/production/xgb_churn_v1.joblib")

# Load Resources
try:
    model = joblib.load(MODEL_PATH)
    # Initialize SHAP Explainer (This runs once at startup)
    explainer = shap.TreeExplainer(model)
    logger.info("✅ Model & SHAP Explainer loaded.")
except Exception as e:
    logger.error(f"Error loading resources: {e}")
    raise RuntimeError("Startup failed")

class CustomerData(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    avg_daily_usage_min: int
    payment_fails_last_3m: int
    last_interaction_sentiment: float

app = FastAPI(title="Sentinel Churn Predictor", version="1.1")

@app.post("/predict_churn")
def predict(data: CustomerData):
    try:
        # Prepare Input
        input_data = data.model_dump()
        input_df = pd.DataFrame([input_data])
        
        # 1. Prediction
        probability = model.predict_proba(input_df)[:, 1][0]
        risk_label = "High Risk" if probability > 0.5 else "Safe"
        
        # 2. Explanation (SHAP)
        shap_values = explainer.shap_values(input_df)
        
        # Create a dictionary of {feature_name: impact_score}
        # Impact > 0 means "Increases Risk", < 0 means "Decreases Risk"
        explanation = dict(zip(input_df.columns, shap_values[0].tolist()))

        return {
            "churn_probability": round(float(probability), 4),
            "risk_label": risk_label,
            "explanation": explanation  # <--- Sending this to Frontend
        }
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))