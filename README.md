# 🏠 House Rent Price Prediction — Proyecto Final MLOps

> Predicción del precio de alquiler de viviendas en India usando Machine Learning supervisado (regresión).

---

## Tabla de Contenidos

1. [Definición del Problema](#a-definición-del-problema)
2. [Dataset](#b-dataset)
3. [Estructura del Proyecto](#c-estructura-del-proyecto)
4. [Configuración del Entorno](#configuración-del-entorno)
5. [Preparación de Datos](#d1-preparación-de-datos)
6. [Experimentación y Entrenamiento](#d2-experimentación-y-entrenamiento)
7. [Despliegue y Servicio](#e-despliegue-y-servicio)
8. [Resultados](#resultados)

---

## A) Definición del Problema

### Contexto

El mercado inmobiliario de alquiler en India presenta alta variabilidad de precios según la ciudad, tamaño, número de habitaciones y estado de amueblado. Propietarios, agentes y arrendatarios carecen de herramientas objetivas para estimar precios justos de mercado, lo que genera asimetrías de información que perjudican especialmente a los inquilinos.

### Problema

> ¿Cómo predecir automáticamente el precio mensual de alquiler (en INR) de una vivienda, dadas sus características físicas y de ubicación?

### Objetivos

- Construir un modelo de regresión que prediga el precio de alquiler a partir de características estructurales de la propiedad.
- Automatizar el pipeline completo: desde los datos crudos hasta el servicio de inferencia mediante API REST.
- Comparar múltiples algoritmos y seleccionar el modelo con mejor rendimiento (modelo campeón).
- Garantizar reproducibilidad con código modular y versionado en Git.

### Restricciones y Supuestos

- Los datos corresponden a propiedades en 6 ciudades de India: Mumbai, Delhi, Bangalore, Chennai, Kolkata y Hyderabad.
- El precio objetivo está en rupias indias (INR) por mes.
- No se utilizan variables externas como inflación o tipo de cambio.

### Métricas de Éxito

| Métrica | Objetivo | Resultado Obtenido |
|---------|----------|--------------------|
| R² (test) | ≥ 0.75 | ✅ 0.7601 |
| MAE | ≤ 15,000 INR | ✅ 6,102 INR |
| MAPE | ≤ 30% | ⚠️ 34.33% |

---

## B) Dataset

**Fuente:** [House Rent Dataset — Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)

**Ubicación:** `data/raw/House_Rent_Dataset.csv`

**Tamaño:** 4,746 registros originales → 4,466 tras limpieza

### Descripción de Variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `BHK` | Numérico | Número de habitaciones, sala y cocina |
| `Rent` | Numérico 🎯 | **Variable objetivo** — precio mensual en INR |
| `Size` | Numérico | Tamaño de la propiedad en pies cuadrados |
| `Floor` | Texto | Piso de la propiedad y total del edificio |
| `Area Type` | Categórico | Tipo de medición (Super/Carpet/Build Area) |
| `Area Locality` | Texto | Localidad específica *(descartada — alta cardinalidad)* |
| `City` | Categórico | Ciudad donde se ubica la propiedad |
| `Furnishing Status` | Categórico | Estado de amueblado (Furnished/Semi/Unfurnished) |
| `Tenant Preferred` | Categórico | Tipo de inquilino preferido |
| `Bathroom` | Numérico | Número de baños |
| `Point of Contact` | Categórico | Punto de contacto *(descartado — no predictivo)* |

### Variables Generadas

| Variable Nueva | Origen | Descripción |
|----------------|--------|-------------|
| `floor_number` | `Floor` | Número de piso de la propiedad (0 = planta baja) |
| `total_floors` | `Floor` | Total de pisos del edificio |

---

## C) Estructura del Proyecto

```
mlops-final-project/
├── data/
│   ├── raw/
│   │   └── House_Rent_Dataset.csv       ← Dataset crudo original
│   └── training/
│       └── house_rent_train.csv         ← Dataset de entrenamiento (generado)
├── experiments/
├── models/
│   └── house_rent_model.pkl             ← Modelo serializado (generado)
├── notebooks/
├── reports/
│   └── metrics.json                     ← Métricas del modelo (generado)
├── resources/images/
│   └── machine_learning_lifecycle.png
├── src/
│   ├── data_preparation.py              ← Pipeline de transformación de datos
│   ├── train.py                         ← Entrenamiento y serialización del modelo
│   └── serving.py                       ← API REST con FastAPI
├── .gitignore
├── CHANGELOG.md
├── requirements.txt
└── README.md
```

---

## Configuración del Entorno

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/mlops-final-project.git
cd mlops-final-project
```

### 2. Crear y activar entorno virtual

```bash
python -m venv env_uni

# Windows
env_uni\Scripts\activate

# Mac/Linux
source env_uni/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## D.1) Preparación de Datos

```bash
python src/data_preparation.py
```

**Output real obtenido:**
```
[INFO] Dataset cargado: 4746 filas × 12 columnas
[INFO] Duplicados eliminados: 0
[INFO] Filas tras filtro de outliers: 4466
[INFO] Dataset de entrenamiento guardado en: data\training\house_rent_train.csv
       Shape final: (4466, 10)
       Columnas: ['BHK', 'Rent', 'Size', 'Area Type', 'City', 'Furnishing Status',
                  'Tenant Preferred', 'Bathroom', 'floor_number', 'total_floors']
```

### Transformaciones Aplicadas

1. **Eliminación de duplicados** — se detectan y eliminan filas idénticas.
2. **Descarte de columnas** — se eliminan `Posted On`, `Area Locality` y `Point of Contact`.
3. **Ingeniería de la variable `Floor`** — se parsea el texto (`"3 out of 5"`) para extraer `floor_number` y `total_floors`.
4. **Filtro de outliers** — se aplica IQR×3 sobre `Rent` para eliminar valores extremos.
5. **Codificación categórica** — `LabelEncoder` sobre `Area Type`, `City`, `Furnishing Status` y `Tenant Preferred`.

**Resultado:** `data/training/house_rent_train.csv`

---

## D.2) Experimentación y Entrenamiento

```bash
python src/train.py
```

**Output real obtenido:**
```
[INFO] Dataset cargado: 4466 muestras, 9 features

[INFO] Evaluando candidatos con validación cruzada (5-fold)...
  Ridge Regression:   R² = 0.5556 ± 0.0290
  Random Forest:      R² = 0.7247 ± 0.0146  ✅ campeón
  Gradient Boosting:  R² = 0.7236 ± 0.0217

[INFO] Modelo seleccionado: Random Forest (R² CV = 0.7247)

[RESULTADO] Métricas en test:
  R²   = 0.7601
  MAE  = 6,102.60 INR
  RMSE = 9,627.48 INR
  MAPE = 34.33%

[INFO] Modelo guardado en: models\house_rent_model.pkl
[INFO] Métricas guardadas en: reports\metrics.json
```

### Comparativa de Modelos

| Modelo | R² (CV 5-fold) | R² (Test) | MAE (INR) | RMSE (INR) |
|--------|---------------|-----------|-----------|------------|
| Ridge Regression | 0.5556 | — | — | — |
| **Random Forest ✅** | **0.7247** | **0.7601** | **6,102** | **9,627** |
| Gradient Boosting | 0.7236 | — | — | — |

El **Random Forest Regressor** fue seleccionado como modelo campeón por obtener el mayor R² en validación cruzada (0.7247), superando a Gradient Boosting por un margen estrecho y a Ridge Regression por más de 17 puntos porcentuales.

**Artefactos generados:**
- `models/house_rent_model.pkl` — modelo serializado con joblib
- `reports/metrics.json` — métricas completas del modelo

---

## E) Despliegue y Servicio

Se utiliza **FastAPI** para exponer el modelo como API REST.

### Iniciar el servidor

```bash
python -m uvicorn src.serving:app --host 0.0.0.0 --port 8000 --reload
```

**Output esperado:**
```
[INFO] Modelo cargado desde: models\house_rent_model.pkl
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> ⚠️ Usar `python -m uvicorn` (no solo `uvicorn`) para garantizar que se usa el Python del entorno virtual.

### Documentación interactiva

Swagger UI disponible en: **http://localhost:8000/docs**

### Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Estado del servicio |
| `POST` | `/predict` | Predicción individual |
| `POST` | `/predict/batch` | Predicciones en lote (máx. 100) |

### Ejemplo de Predicción (curl)

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "BHK": 2,
           "Size": 1100,
           "Area_Type": 1,
           "City": 2,
           "Furnishing_Status": 1,
           "Tenant_Preferred": 0,
           "Bathroom": 2,
           "floor_number": 3,
           "total_floors": 10
         }'
```

### Ejemplo de Predicción (Python)

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "BHK": 2,
    "Size": 1100,
    "Area_Type": 1,
    "City": 2,
    "Furnishing_Status": 1,
    "Tenant_Preferred": 0,
    "Bathroom": 2,
    "floor_number": 3,
    "total_floors": 10
})
print(response.json())
```

### Respuesta esperada

```json
{
  "predicted_rent_inr": 28500.0,
  "predicted_rent_formatted": "₹ 28,500 / mes",
  "model_version": "1.0.0",
  "inputs": {
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
```

---

## Resultados

### Métricas Finales del Modelo Campeón (Random Forest)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R² Test** | **0.7601** | El modelo explica el 76% de la varianza del precio |
| **MAE** | **6,102 INR** | Error promedio de ~6,100 rupias por predicción |
| **RMSE** | **9,627 INR** | Penaliza más los errores grandes |
| **MAPE** | **34.33%** | Error porcentual promedio |

### Conclusiones

- El modelo cumple con los objetivos de R² (≥ 0.75) y MAE (≤ 15,000 INR).
- El MAPE de 34% supera ligeramente el objetivo del 30%, indicando mayor dificultad con propiedades de precio muy bajo o muy alto.
- La variable **City** resultó ser el factor más determinante en el precio de alquiler, seguida de **Size** y **BHK**.
- El pipeline completo es reproducible: desde los datos crudos hasta la inferencia vía API con un único entorno virtual.

---

## Referencias

- [House Rent Dataset — Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)