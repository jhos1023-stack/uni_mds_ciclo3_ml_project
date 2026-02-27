#%%
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#%%
TRAIN_PATH = os.path.join("data", "training", "house_rent_train.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_rent_model.pkl")
METRICS_PATH = os.path.join("reports", "metrics.json")

TARGET = "Rent"
FEATURES = [
    "BHK", "Size", "Area Type", "City",
    "Furnishing Status", "Tenant Preferred",
    "Bathroom", "floor_number", "total_floors"
]


# ──────────────────────────────────────────────
# 1. Carga y split
# ──────────────────────────────────────────────
def load_dataset(path: str):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    print(f"[INFO] Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} features")
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ──────────────────────────────────────────────
# 2. Selección del mejor modelo
# ──────────────────────────────────────────────
def select_best_model(X_train, y_train) -> tuple:
    candidates = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }

    print("\n[INFO] Evaluando candidatos con validación cruzada (5-fold)...")
    best_name, best_score, best_model = None, -np.inf, None
    cv_results = {}

    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        mean_r2 = scores.mean()
        cv_results[name] = {"r2_mean": round(mean_r2, 4), "r2_std": round(scores.std(), 4)}
        print(f"  {name}: R² = {mean_r2:.4f} ± {scores.std():.4f}")
        if mean_r2 > best_score:
            best_score, best_name, best_model = mean_r2, name, model

    print(f"\n[INFO] Modelo seleccionado: {best_name} (R² CV = {best_score:.4f})")
    return best_name, best_model, cv_results


# ──────────────────────────────────────────────
# 3. Entrenamiento y evaluación
# ──────────────────────────────────────────────
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name: str, cv_results: dict) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

    metrics = {
        "model": model_name,
        "test_r2": round(r2, 4),
        "test_mae": round(mae, 2),
        "test_rmse": round(rmse, 2),
        "test_mape_pct": round(mape, 2),
        "cv_results": cv_results,
        "features": FEATURES,
        "target": TARGET,
    }

    print(f"\n[RESULTADO] Métricas en test:")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAE  = {mae:,.2f} INR")
    print(f"  RMSE = {rmse:,.2f} INR")
    print(f"  MAPE = {mape:.2f}%")

    return metrics


# ──────────────────────────────────────────────
# 4. Persistencia
# ──────────────────────────────────────────────
def save_artifacts(model, metrics: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"\n[INFO] Modelo guardado en: {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Métricas guardadas en: {METRICS_PATH}")


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────
def run_training():
    X_train, X_test, y_train, y_test = load_dataset(TRAIN_PATH)
    model_name, model, cv_results = select_best_model(X_train, y_train)
    metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, cv_results)
    save_artifacts(model, metrics)
    return model, metrics


if __name__ == "__main__":
    run_training()