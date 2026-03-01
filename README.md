# 🏠 House Rent Price Prediction — Proyecto Final MLOps

> Predicción del precio de alquiler de viviendas en India usando Machine Learning supervisado (regresión).

---

## Tabla de Contenidos

1. [Definición del Problema](#a-definición-del-problema)
2. [Dataset y Variables](#b-dataset-y-variables)
3. [Estructura del Proyecto](#c-estructura-del-proyecto)
4. [Configuración del Entorno](#configuración-del-entorno)
5. [Preparación de Datos](#d1-preparación-de-datos)
6. [Experimentación y Entrenamiento](#d2-experimentación-y-entrenamiento)
7. [Despliegue y Servicio](#e-despliegue-y-servicio)
8. [Resultados y Predicciones](#f-resultados-y-predicciones)
9. [Conclusiones, Insights y Lecciones Aprendidas](#g-conclusiones-insights-y-lecciones-aprendidas)

---

## A) Definición del Problema

### Contexto

El mercado inmobiliario de alquiler en India es uno de los más dinámicos y complejos de Asia. Con más de 1,400 millones de habitantes y una urbanización acelerada, la demanda de vivienda en alquiler crece constantemente en ciudades como Mumbai, Delhi y Bangalore. Sin embargo, los precios varían enormemente según factores como la ciudad, el tamaño, el número de habitaciones, la ubicación y el estado de amueblado.

Esta variabilidad crea una asimetría de información: propietarios y agentes tienen ventaja sobre los arrendatarios, quienes muchas veces no tienen referencias objetivas para evaluar si un precio es justo. Del mismo modo, los propietarios pueden tener dificultad para fijar precios competitivos sin perder rentabilidad.

### Problema

> ¿Cómo predecir automáticamente el precio mensual de alquiler (en INR) de una vivienda en India, dadas sus características físicas y de ubicación?

### Objetivos

- Construir un modelo de regresión supervisado que prediga el precio mensual de alquiler a partir de características de la propiedad.
- Automatizar el pipeline completo: ingesta de datos crudos → transformación → entrenamiento → servicio de inferencia vía API REST.
- Comparar múltiples algoritmos de ML y seleccionar el modelo campeón mediante validación cruzada.
- Garantizar reproducibilidad total con código modular, entorno virtual y versionado en Git.

### Restricciones y Supuestos

- Los datos corresponden a propiedades en 6 ciudades de India: Mumbai, Delhi, Bangalore, Chennai, Kolkata y Hyderabad.
- El precio objetivo está en rupias indias (INR) por mes.
- No se utilizan variables externas dinámicas como inflación, tipo de cambio o índices de mercado.
- El modelo es descriptivo del dataset disponible y puede no generalizar perfectamente a otras regiones o períodos temporales.

### Métricas de Éxito

| Métrica | Descripción | Objetivo | Resultado |
|---------|-------------|----------|-----------|
| R² (test) | Varianza explicada por el modelo | ≥ 0.75 | ✅ 0.7601 |
| MAE | Error absoluto medio en INR | ≤ 15,000 INR | ✅ 6,102 INR |
| MAPE | Error absoluto porcentual medio | ≤ 30% | ⚠️ 34.33% |

---

## B) Dataset y Variables

**Fuente:** [House Rent Dataset — Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)

**Ubicación en el repo:** `data/raw/House_Rent_Dataset.csv`

**Tamaño original:** 4,746 registros × 12 columnas

**Tamaño tras limpieza:** 4,466 registros × 10 columnas

### Variables del Dataset Original

| Variable | Tipo | Descripción | Uso en Modelo |
|----------|------|-------------|---------------|
| `BHK` | Numérico | Número de habitaciones, sala y cocina | ✅ Feature |
| `Rent` | Numérico | **Variable objetivo** — precio mensual en INR | 🎯 Target |
| `Size` | Numérico | Tamaño de la propiedad en pies cuadrados | ✅ Feature |
| `Floor` | Texto | Piso actual y total de pisos del edificio | ✅ Transformada |
| `Area Type` | Categórico | Tipo de medición del área (Super/Carpet/Build) | ✅ Feature |
| `Area Locality` | Texto | Localidad específica | ❌ Descartada (alta cardinalidad) |
| `City` | Categórico | Ciudad de la propiedad | ✅ Feature |
| `Furnishing Status` | Categórico | Estado de amueblado | ✅ Feature |
| `Tenant Preferred` | Categórico | Tipo de inquilino preferido | ✅ Feature |
| `Bathroom` | Numérico | Número de baños | ✅ Feature |
| `Point of Contact` | Categórico | Punto de contacto del anuncio | ❌ Descartada (no predictiva) |
| `Posted On` | Fecha | Fecha de publicación | ❌ Descartada (no predictiva) |

### Variables Generadas en la Preparación

| Variable Nueva | Origen | Transformación | Descripción |
|----------------|--------|----------------|-------------|
| `floor_number` | `Floor` | Regex parsing | Piso de la propiedad (0 = planta baja, -1 = sótano) |
| `total_floors` | `Floor` | Regex parsing | Total de pisos del edificio |

### Dataset Final de Entrenamiento

El dataset de entrenamiento contiene **9 features** y **1 variable objetivo**:

```
Columnas: BHK, Size, Area Type, City, Furnishing Status,
          Tenant Preferred, Bathroom, floor_number, total_floors
Target:   Rent
```

### Codificación de Variables Categóricas

| Variable | Valores originales | Codificación |
|----------|--------------------|--------------|
| `Area Type` | Build Area, Carpet Area, Super Area | 0, 1, 2 |
| `City` | Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai | 0, 1, 2, 3, 4, 5 |
| `Furnishing Status` | Furnished, Semi-Furnished, Unfurnished | 0, 1, 2 |
| `Tenant Preferred` | Bachelors, Bachelors/Family, Family | 0, 1, 2 |

---

## C) Estructura del Proyecto

```
mlops-final-project/
├── data/
│   ├── raw/
│   │   └── House_Rent_Dataset.csv       ← Dataset crudo original
│   └── training/
│       └── house_rent_train.csv         ← Dataset de entrenamiento (generado)
├── experiments/                          ← Espacio para experimentos MLflow
├── models/
│   └── house_rent_model.pkl             ← Modelo Random Forest serializado
├── notebooks/                            ← Notebooks de exploración EDA
├── reports/
│   └── metrics.json                     ← Métricas del modelo campeón
├── resources/images/
│   └── machine_learning_lifecycle.png
├── src/
│   ├── data_preparation.py              ← Pipeline de transformación de datos
│   ├── train.py                         ← Entrenamiento y serialización
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
git clone https://github.com/jhos1023-stack/uni_mds_ciclo3_ml_project.git
cd uni_mds_ciclo3_ml_project
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

El script `src/data_preparation.py` implementa el pipeline completo de limpieza y transformación.

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

**1. Eliminación de duplicados**
Se detectan y eliminan filas completamente idénticas. En este dataset no se encontraron duplicados (0 eliminados), lo que indica buena calidad en la recolección original.

**2. Descarte de columnas de baja utilidad**
Se eliminaron `Posted On`, `Area Locality` y `Point of Contact`. `Area Locality` fue descartada a pesar de ser potencialmente informativa, porque tiene demasiados valores únicos (alta cardinalidad) que dificultarían la generalización del modelo sin técnicas de codificación avanzadas como target encoding.

**3. Ingeniería de la variable `Floor`**
La columna original contiene texto libre como `"3 out of 5"`, `"Ground out of 2"` o `"Upper Basement"`. Se aplicó parsing con expresiones regulares para extraer dos variables numéricas: `floor_number` y `total_floors`. Los valores `"Ground"` se mapearon a 0 y los valores de sótano a -1.

**4. Filtro de outliers sobre `Rent`**
Se aplicó el método del Rango Intercuartílico (IQR) con factor ×3 sobre la variable objetivo. Esto eliminó 280 registros (5.9% del total) con rentas extremadamente altas o bajas que distorsionarían el entrenamiento. Se eligió factor ×3 en lugar del estándar ×1.5 para ser conservador y no perder demasiados datos.

**5. Codificación de variables categóricas**
Se aplicó `LabelEncoder` de scikit-learn a las 4 variables categóricas. Esta codificación es adecuada para modelos basados en árboles (como Random Forest) que no asumen relaciones ordinales entre categorías.

**Resultado:** `data/training/house_rent_train.csv` (4,466 registros, 10 columnas)

---

## D.2) Experimentación y Entrenamiento

El script `src/train.py` implementa la selección automática del modelo campeón mediante validación cruzada.

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

### Estrategia de Experimentación

Se evaluaron 3 modelos candidatos con validación cruzada de 5 folds, usando R² como métrica de selección. El dataset se dividió en 80% entrenamiento y 20% test con semilla aleatoria fija (random_state=42) para garantizar reproducibilidad.

### Comparativa Detallada de Modelos

| Modelo | R² CV (media) | R² CV (std) | R² Test | MAE (INR) | RMSE (INR) |
|--------|--------------|-------------|---------|-----------|------------|
| Ridge Regression | 0.5556 | ±0.0290 | — | — | — |
| **Random Forest ✅** | **0.7247** | **±0.0146** | **0.7601** | **6,102** | **9,627** |
| Gradient Boosting | 0.7236 | ±0.0217 | — | — | — |

### Análisis de la Selección del Modelo Campeón

**Random Forest** fue seleccionado como modelo campeón por tres razones:

1. **Mayor R² en CV (0.7247)** — supera a Gradient Boosting por 0.0011 y a Ridge por 0.1691.
2. **Menor desviación estándar (±0.0146)** — es el modelo más estable entre los folds, lo que indica mejor generalización.
3. **Sin necesidad de escalado** — a diferencia de Ridge, Random Forest funciona directamente con los datos codificados sin necesidad de normalización.

**Parámetros del modelo campeón:**
```python
RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
```

**Artefactos generados:**
- `models/house_rent_model.pkl` — modelo serializado con joblib
- `reports/metrics.json` — métricas completas del modelo campeón

---

## E) Despliegue y Servicio

Se utiliza **FastAPI** para exponer el modelo como API REST. FastAPI fue elegido sobre Flask por su validación automática de datos con Pydantic, documentación Swagger generada automáticamente y mejor rendimiento asíncrono.

### Iniciar el servidor

```bash
python -m uvicorn src.serving:app --host 0.0.0.0 --port 8000 --reload
```

> ⚠️ Usar `python -m uvicorn` (no solo `uvicorn`) para garantizar que se usa el Python del entorno virtual activo.

**Output esperado:**
```
[INFO] Modelo cargado desde: models\house_rent_model.pkl
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Documentación Interactiva

Swagger UI disponible en: **http://localhost:8000/docs**

### Endpoints Disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Health check y estado del servicio |
| `GET` | `/health` | Estado simplificado |
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

### Respuesta de la API

```json
{
  "predicted_rent_inr": 28500.0,
  "predicted_rent_formatted": "Rs. 28,500 / mes",
  "version": "1.0.0",
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

## F) Resultados y Predicciones

### Métricas Finales del Modelo Campeón

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R² Test** | **0.7601** | El modelo explica el 76% de la varianza del precio de alquiler |
| **MAE** | **6,102 INR** | En promedio, el modelo se equivoca por ~6,100 rupias por predicción |
| **RMSE** | **9,627 INR** | Penaliza más los errores grandes; errores de ~9,600 INR en promedio cuadrático |
| **MAPE** | **34.33%** | El error porcentual promedio es del 34%, mayor en propiedades de precio extremo |

### Ejemplos de Predicciones

| BHK | Size (ft²) | Ciudad | Amueblado | Precio Real | Predicción | Error |
|-----|-----------|--------|-----------|-------------|------------|-------|
| 1 | 500 | Kolkata (4) | No (2) | ~7,000 INR | ~8,200 INR | ~17% |
| 2 | 1100 | Delhi (2) | Semi (1) | ~28,000 INR | ~27,500 INR | ~2% |
| 3 | 1800 | Mumbai (5) | Sí (0) | ~65,000 INR | ~58,000 INR | ~11% |
| 4 | 3000 | Bangalore (0) | Sí (0) | ~120,000 INR | ~98,000 INR | ~18% |

### Insights Clave del Modelo

**1. Ciudad es el factor más determinante**
La variable `City` tiene el mayor peso en las predicciones. Mumbai y Delhi tienen precios promedio significativamente más altos que Kolkata y Chennai. El modelo captura bien esta diferencia geográfica.

**2. El tamaño importa, pero no linearmente**
`Size` es la segunda variable más importante. Sin embargo, la relación no es perfectamente lineal: propiedades muy grandes en ciudades secundarias no necesariamente alcanzan precios proporcionales a su tamaño.

**3. Amueblado tiene impacto considerable**
Las propiedades completamente amuebladas (`Furnished`) tienen un premium de precio significativo frente a las no amuebladas, especialmente en Mumbai y Bangalore donde el mercado de expatriados es activo.

**4. El número de habitaciones (BHK) correlaciona con el precio**
A mayor BHK, mayor precio esperado, pero la correlación es moderada porque depende fuertemente de la ciudad y el tamaño real de la propiedad.

**5. El piso tiene impacto menor al esperado**
`floor_number` y `total_floors` resultaron ser variables de menor importancia relativa, posiblemente porque en India los pisos altos no siempre tienen el premium que tienen en mercados occidentales.

---

## G) Conclusiones, Insights y Lecciones Aprendidas

### Conclusiones

- **El modelo cumple los objetivos principales:** R² de 0.76 (objetivo ≥ 0.75) y MAE de 6,102 INR (objetivo ≤ 15,000 INR). Esto significa que el modelo es útil como referencia de precios de mercado.

- **El MAPE del 34% supera el objetivo del 30%**, lo que indica que el modelo tiene dificultades con propiedades en los extremos del rango de precios. Propiedades muy baratas (< 5,000 INR) o muy caras (> 100,000 INR) son más difíciles de predecir con precisión porcentual.

- **Random Forest superó a Gradient Boosting** a pesar de que en teoría Gradient Boosting suele rendir mejor. Esto puede deberse al tamaño relativamente pequeño del dataset (4,466 registros) donde Random Forest es más estable.

- **La preparación de datos fue la etapa más crítica.** El parsing de la columna `Floor` y el manejo correcto de los outliers impactaron significativamente en la calidad del modelo final.

- **El pipeline end-to-end funciona correctamente:** desde los datos crudos en CSV hasta una predicción via API REST en producción, todo el proceso es automatizable y reproducible.

### Limitaciones

- **Dataset pequeño (4,746 registros):** Para un problema de precio de bienes raíces, este volumen es limitado. Con más datos el modelo podría generalizar mejor.
- **Datos de un único período temporal:** No se conoce el período exacto de recolección. Los precios de alquiler cambian con el tiempo y el modelo puede quedar desactualizado.
- **`Area Locality` descartada:** Esta variable podría ser muy predictiva (el barrio específico importa mucho en bienes raíces), pero su alta cardinalidad requiere técnicas más avanzadas.
- **Sin variables externas:** Factores como proximidad a transporte público, escuelas, hospitales o zonas comerciales no están incluidos y tienen alto impacto en el precio real.
- **MAPE elevado en extremos:** El modelo predice bien el rango medio de precios pero tiene mayor error en propiedades muy baratas o muy caras.

### Mejoras Futuras

- **Incorporar `Area Locality`** usando técnicas de codificación como Target Encoding o embeddings para capturar la información del barrio sin el problema de alta cardinalidad.
- **Aumentar el dataset** con datos más recientes de Kaggle u otras fuentes para mejorar la generalización temporal.
- **Usar modelos más avanzados** como XGBoost, LightGBM o CatBoost, que suelen rendir mejor en datos tabulares con variables categóricas.
- **Hyperparameter tuning** con GridSearchCV o Optuna para optimizar los parámetros del Random Forest más allá de los valores por defecto.
- **Feature engineering adicional:** ratios como precio por pie cuadrado, interacciones entre ciudad y tamaño, o clusters de localidad.
- **Monitoreo del modelo en producción** con MLflow o herramientas de data drift para detectar cuando el modelo se desactualiza.
- **Containerización con Docker** para hacer el deployment más robusto y portable.

### Lecciones Aprendidas

1. **Los entornos virtuales son esenciales desde el inicio.** No haberlo configurado desde el principio generó conflictos de versiones entre Python 3.7, 3.9 y scikit-learn que costaron tiempo valioso de debugging.

2. **Las versiones de las librerías importan mucho.** Un modelo entrenado con scikit-learn 1.6.1 no puede cargarse con scikit-learn 1.0.2 (pickle protocol incompatible). El entorno donde se entrena y donde se sirve el modelo deben ser idénticos.

3. **`python -m uvicorn` vs `uvicorn` no es lo mismo en Windows.** Cuando hay múltiples instalaciones de Python, llamar directamente a `uvicorn` puede usar una versión diferente a la del entorno virtual activo.

4. **La ingeniería de datos consume más tiempo que el modelado.** El parsing de la columna `Floor`, el manejo de outliers y la decisión sobre qué variables descartar tomaron más esfuerzo que la selección y entrenamiento del modelo en sí.

5. **FastAPI es superior a Flask para APIs de ML** en términos de validación automática, documentación y facilidad de testing con Swagger UI integrado.

 
---

## Referencias

- [House Rent Dataset — Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Random Forest Regressor — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/)