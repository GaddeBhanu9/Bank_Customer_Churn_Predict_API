from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI
app = FastAPI(title="Bank Churn Predictor")

# Global variables
artifacts = None

@app.on_event("startup")
async def load_artifacts():
    global artifacts
    artifacts = joblib.load("churn_model.pkl")
    print("Model, Scaler, Encoders, and Feature Names loaded successfully!")

# ---------- PYDANTIC SCHEMA (Raw Inputs) ----------
class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850)
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=0, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)
    
    Geography: str
    Gender: str

    # Validators to catch bad inputs before they hit the encoder
    @field_validator('Geography')
    def validate_geography(cls, v):
        allowed = ['France', 'Spain', 'Germany']
        if v not in allowed:
            raise ValueError(f'Geography must be one of {allowed}')
        return v

    @field_validator('Gender')
    def validate_gender(cls, v):
        allowed = ['Male', 'Female']
        if v not in allowed:
            raise ValueError(f'Gender must be one of {allowed}')
        return v

# ---------- PREDICTION ENDPOINT ----------
@app.post("/predict")
async def predict(customer: CustomerInput):
    try:
        # 1. Convert to DataFrame
        df = pd.DataFrame([customer.dict()])
        
        # 2. Apply Label Encoders
        df['Gender'] = artifacts['gender_encoder'].transform(df['Gender'])
        df['Geography'] = artifacts['geo_encoder'].transform(df['Geography'])
        
        # 3. Select features in the exact order used during training
        X = df[artifacts['feature_names']].copy()
        
        # 4. Apply Standard Scaler
        X_scaled = artifacts['scaler'].transform(X)
        
        # 5. Predict!
        prediction = artifacts['model'].predict(X_scaled)
        probability = artifacts['model'].predict_proba(X_scaled)[0][1]
        
        return {
            "churn_prediction": int(prediction[0]),
            "churn_probability": round(float(probability), 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- HEALTH CHECK ----------
@app.get("/")
async def root():
    return {"message": "Churn Prediction API is running. Use POST /predict"}