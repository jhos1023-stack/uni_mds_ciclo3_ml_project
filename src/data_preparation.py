#%%

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

RAW_PATH = os.path.join("data", "raw", "House_Rent_Dataset.csv")
TRAIN_PATH = os.path.join("data", "training", "house_rent_train.csv")

#%%
# ──────────────────────────────────────────────
# 1. Carga
# ──────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


# ──────────────────────────────────────────────
# 2. Limpieza
# ──────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminar duplicados
    antes = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Duplicados eliminados: {antes - len(df)}")

    # Eliminar columnas de baja utilidad para el modelo
    drop_cols = ["Posted On", "Area Locality", "Point of Contact"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Extraer el número de piso actual y el total de pisos desde la columna 'Floor'
    def parse_floor(val):
        if pd.isna(val):
            return 0, 1
        val = str(val).lower()
        match = re.search(r"(\d+)\s+out\s+of\s+(\d+)", val)
        if match:
            return int(match.group(1)), int(match.group(2))
        if "ground" in val:
            total = re.search(r"out\s+of\s+(\d+)", val)
            return 0, int(total.group(1)) if total else 1
        if "upper basement" in val or "lower basement" in val:
            return -1, 1
        return 0, 1

    df[["floor_number", "total_floors"]] = df["Floor"].apply(
        lambda x: pd.Series(parse_floor(x))
    )
    df.drop(columns=["Floor"], inplace=True)

    # Eliminar rentas atípicas (outliers) usando IQR
    Q1 = df["Rent"].quantile(0.25)
    Q3 = df["Rent"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Rent"] >= Q1 - 3 * IQR) & (df["Rent"] <= Q3 + 3 * IQR)]
    print(f"[INFO] Filas tras filtro de outliers: {len(df)}")

    return df


# ──────────────────────────────────────────────
# 3. Codificación de variables categóricas
# ──────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = ["Area Type", "City", "Furnishing Status", "Tenant Preferred"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


# ──────────────────────────────────────────────
# 4. Guardar dataset de entrenamiento
# ──────────────────────────────────────────────
def save_dataset(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Dataset de entrenamiento guardado en: {path}")
    print(f"       Shape final: {df.shape}")
    print(f"       Columnas: {list(df.columns)}")


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────
def run_pipeline():
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = encode_features(df)
    save_dataset(df, TRAIN_PATH)
    return df


if __name__ == "__main__":
    run_pipeline()