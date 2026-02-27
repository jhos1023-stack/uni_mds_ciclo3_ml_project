 
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join("..","models", "house_rent_model.pkl")

app = FastAPI(
    title="House Rent Prediction API",
    description=(
        "API para predecir el precio de alquiler de viviendas en India "
        "a partir de características como ubicación, tamaño y amueblado."
    ),
    version="1.0.0",
)

# Cargar modelo al iniciar
try:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Modelo cargado desde: {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"[WARN] Modelo no encontrado en {MODEL_PATH}. Ejecuta train.py primero.")


# ──────────────────────────────────────────────
# Esquemas de entrada/salida
# ──────────────────────────────────────────────
class HouseFeatures(BaseModel):
    BHK: int = Field(..., ge=1, le=10, description="Número de habitaciones (BHK)")
    Size: float = Field(..., gt=0, description="Tamaño en pies cuadrados")
    Area_Type: int = Field(..., ge=0, le=2, description="Tipo de área (0=Build, 1=Carpet, 2=Super)")
    City: int = Field(..., ge=0, le=5, description="Ciudad codificada (0-5)")
    Furnishing_Status: int = Field(..., ge=0, le=2, description="0=Unfurnished, 1=Semi, 2=Furnished")
    Tenant_Preferred: int = Field(..., ge=0, le=2, description="Tipo de inquilino preferido")
    Bathroom: int = Field(..., ge=1, le=10, description="Número de baños")
    floor_number: int = Field(0, description="Piso de la propiedad (0=planta baja)")
    total_floors: int = Field(1, ge=1, description="Total de pisos del edificio")

    class Config:
        json_schema_extra = {
            "example": {
                "BHK": 2,
                "Size": 1100,
                "Area_Type": 1,
                "City": 2,
                "Furnishing_Status": 1,
                "Tenant_Preferred": 0,
                "Bathroom": 2,
                "floor_number": 3,
                "total_floors": 10
            }
        }


class PredictionResponse(BaseModel):
    predicted_rent_inr: float
    predicted_rent_formatted: str
    model_version: str = "1.0.0"
    inputs: dict


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {
        "status": "ok",
        "message": "House Rent Prediction API está activa.",
        "model_loaded": model is not None,
        "docs": "/docs"
    }


@app.get("/health", summary="Estado del servicio")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, summary="Predecir alquiler")
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta src/train.py para entrenar el modelo."
        )

    feature_names = [
        "BHK", "Size", "Area Type", "City",
        "Furnishing Status", "Tenant Preferred",
        "Bathroom", "floor_number", "total_floors"
    ]

    # Mapear nombres con espacios al orden esperado por el modelo
    X = np.array([[
        features.BHK,
        features.Size,
        features.Area_Type,
        features.City,
        features.Furnishing_Status,
        features.Tenant_Preferred,
        features.Bathroom,
        features.floor_number,
        features.total_floors,
    ]])

    prediction = float(model.predict(X)[0])
    prediction = max(0, prediction)  # No permitir valores negativos

    return PredictionResponse(
        predicted_rent_inr=round(prediction, 2),
        predicted_rent_formatted=f"₹ {prediction:,.0f} / mes",
        inputs=features.dict()
    )


@app.post("/predict/batch", summary="Predicciones en lote")
def predict_batch(features_list: List[HouseFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    if len(features_list) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 predicciones por lote.")

    results = []
    for features in features_list:
        X = np.array([[
            features.BHK, features.Size, features.Area_Type,
            features.City, features.Furnishing_Status, features.Tenant_Preferred,
            features.Bathroom, features.floor_number, features.total_floors,
        ]])
        pred = float(model.predict(X)[0])
        results.append({
            "predicted_rent_inr": round(max(0, pred), 2),
            "predicted_rent_formatted": f"₹ {max(0, pred):,.0f} / mes"
        })
    return {"predictions": results, "count": len(results)}