# Pipeline de Airflow - SodAI Drinks

## Descripción General

Pipeline de MLOps que automatiza el ciclo de vida del modelo predictivo: ingesta de datos, preprocesamiento, detección de drift, reentrenamiento condicional y generación de predicciones. Utiliza Apache Airflow para orquestación y MLflow para tracking de experimentos

## Arquitectura del DAG

El DAG `sodai_ml_pipeline` consta de 7 tareas con branching condicional:

1. **ingestar_datos**: Carga parquets de transacciones, productos y clientes
2. **preprocesar_datos**: Pipeline de 14 pasos (merge, features temporales, geo-clustering, encoding, scaling). Genera `pipeline_pp.pkl`
3. **detectar_drift**: Verifica existencia de modelo y aplica test de Kolmogorov-Smirnov (p-value < 0.05). Decide entre reentrenar o usar modelo existente
4. **reentrenar_modelo**: Entrena DecisionTreeClassifier. Si no hay modelo previo usa Optuna (30 trials), si ya existe usa parámetros fijos. Tracking en MLflow con métricas y SHAP plots
5. **usar_modelo_existente**: Carga el modelo más reciente si no hay drift
6. **generar_predicciones**: Crea predicciones para la semana siguiente usando el pipeline y modelo
7. **fin_pipeline**: Marca fin exitoso del pipeline

## Interfaz de Airflow

Accede a http://localhost:8080 (usuario: `airflow`, password: `airflow`) para visualizar el DAG y sus ejecuciones. En la vista Graph verás el flujo completo con branching condicional.

## Estructura de Directorios

```
airflow/
├── dags/
│   ├── pipeline.py              # Definición del DAG
│   └── helper_functions.py      # Funciones ML y transformers
├── data/
│   ├── raw/                     # Datos de entrada (parquets)
│   ├── models/                  # Modelos (.pkl)
│   └── predictions/             # Predicciones (.csv)
├── mlruns/                      # Experimentos MLflow
└── docker-compose.yml           # Orquestación de servicios
```

## Helper Functions del Backend (`dags/helper_functions.py`)

El archivo `helper_functions.py` contiene todas las funciones de ML y transformadores personalizados utilizados por el pipeline:

### Clases Transformadoras Personalizadas

1. **`GeoClustering`**: Clustering geográfico basado en coordenadas X, Y usando KMeans
   - Agrupa clientes por ubicación geográfica
   - Genera feature `geo_cluster` (4 clusters por defecto)
   - Compatible con sklearn Pipeline

2. **`IQR`**: Eliminación de outliers usando método IQR
   - Reemplaza valores extremos con NaN (no elimina filas)
   - Factor configurable (default: 1.5)
   - Preserva estructura del dataset

3. **`FeatureAggregator`**: Genera features temporales y de frecuencia
   - Frecuencia de compra por producto, categoría y marca
   - Feature de trimestre
   - Requiere `df_transacciones` como parámetro en `transform()`

### Funciones Principales

1. **`load_data()`**: Carga datos desde `/data/raw/`
   - `transacciones.parquet`: Histórico de compras
   - `clientes.parquet`: Información de clientes
   - `productos.parquet`: Catálogo de productos

2. **`preprocess_data()`**: Pipeline completo de preprocesamiento (14 pasos)
   - Conversión de tipos de datos
   - Agrupación y suma de items
   - Filtrado de valores negativos
   - Creación de variables temporales (Semana, Año)
   - Generación de target binario
   - Merge de transacciones, clientes y productos
   - Split temporal (70% train, 15% val, 15% test)
   - Undersampling con ratio 10:1 (clase 0:clase 1)
   - Pipeline con scaling (MinMaxScaler) y encoding (OneHotEncoder)

3. **`detect_drift()`**: Detección de drift entre datasets
   - **Numéricas**: Test de Kolmogorov-Smirnov (p-value < 0.05)
   - **Categóricas**: Jensen-Shannon Divergence
   - Threshold configurable (default: 0.1)
   - Retorna `True` si se detecta drift significativo

4. **`optimize_with_optuna()`**: Optimización de hiperparámetros
   - Ejecutado solo cuando NO existe modelo previo
   - 30 trials con TPE Sampler
   - Optimiza: `max_depth`, `min_samples_split`, `min_samples_leaf`
   - Métrica objetivo: F1-score macro en validación

5. **`train_and_log_model()`**: Entrenamiento con MLflow tracking
   - Modelo: DecisionTreeClassifier
   - Tracking de hiperparámetros y métricas en MLflow
   - Generación de gráficos SHAP para interpretabilidad
   - Guardado local del modelo (.pkl)
   - Métricas: F1-macro, Accuracy, Precision, Recall por clase

6. **`predict_future_week()`**: Generación de predicciones
   - Predice probabilidad de compra para la próxima semana
   - Guarda predicciones en formato parquet
   - Timestamp automático en nombre de archivo

### Configuración Global

```python
DATA_PATH = "/opt/airflow/data"
MODEL_PATH = "/opt/airflow/data/models"
PRED_PATH = "/opt/airflow/data/predictions"
MLFLOW_TRACKING_URI = "http://mlflow:5001"
MLFLOW_EXPERIMENT = "modelo_tiendas_ancla"
```

## Archivo `.env` (variables de entorno)

El proyecto espera estas variables en `.env` (valores de ejemplo):

```
# Usuario y contraseña de Airflow Web UI
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# UID para permisos de archivos
AIRFLOW_UID=501

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
```


## Ejecución

### Paso 1: Levantar servicios
```bash
cd Proyecto/entrega2/airflow
docker compose up -d
```

### Paso 2: Verificar contenedores
```bash
docker compose ps
# Esperar ~30 segundos para inicialización
```

### Paso 3: Acceder a interfaces
- **Airflow**: http://localhost:8080 (user:airflow/pasword:airflow)
- **MLflow**: http://localhost:5001

### Paso 4: Ejecutar DAG
Desde la interfaz web de Airflow:
1. Activar toggle del DAG `pipeline_modelo_lightgbm`
2. Click en "Trigger DAG"
3. Monitorear en vista "Graph"




### Verificar resultados
```bash
ls data/models/        # modelo_*.pkl, pipeline_pp.pkl, shap_summary.png
ls data/predictions/   # predicciones_*.csv
```

## Probar el Modelo con Gradio

Una vez que el DAG se haya ejecutado exitosamente y generado el modelo, puedes probar el sistema de predicción a través de interfaces Gradio:

### Opción 1: Aplicación Principal (Predicciones Individuales)

```bash
cd ../app
docker compose up -d
```

Accede a:
- **Frontend Gradio**: http://localhost:7860
- **API Backend**: http://localhost:8000/docs

**Funcionalidades**:
- Predecir si un cliente comprará un producto específico
- Ver probabilidad de compra
- Consultar estado del sistema y modelo
- Visualizar información del modelo (hiperparámetros, features)

### Opción 2: Sistema de Recomendación (Top N Productos)

```bash
cd ../bonus/recsys
docker compose up -d
```

Accede a:
- **Frontend Gradio**: http://localhost:7861
- **API Backend**: http://localhost:8001/docs

**Funcionalidades**:
- Generar recomendaciones personalizadas para un cliente
- Top N productos ordenados por probabilidad de compra
- Evalúa todos los productos (~971) automáticamente

### Opción 3: Chatbot Conversacional (Consultas con LLM)

```bash
cd ../bonus/llm
docker compose up -d
```

Accede a:
- **Frontend Gradio**: http://localhost:7862
- **API Backend**: http://localhost:8002/docs

**Funcionalidades**:
- Chatear en lenguaje natural sobre los datos
- Preguntar estadísticas y consultas complejas
- Análisis exploratorio conversacional

**Nota**: Requiere API key de Groq (gratis en [console.groq.com](https://console.groq.com/))

### Requisitos Previos para las Apps

Todas las aplicaciones requieren que el DAG de Airflow haya generado:
- ✅ Modelo entrenado: `data/models/modelo_*.pkl`
- ✅ Pipeline de preprocesamiento: `data/models/pipeline_pp.pkl`
- ✅ Datos en formato parquet: `data/raw/*.parquet`

### Diagrama de Flujo

```
┌─────────────────┐
│  Airflow DAG    │  ← Pipeline de entrenamiento
│  (puerto 8080)  │
└────────┬────────┘
         │ Genera modelos y pipeline
         ├────────────────┬────────────────┬─────────────────┐
         ▼                ▼                ▼                 ▼
┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐
│ app/        │  │ recsys/     │  │ llm/         │  │ MLflow      │
│ (7860/8000) │  │ (7861/8001) │  │ (7862/8002)  │  │ (5001)      │
└─────────────┘  └─────────────┘  └──────────────┘  └─────────────┘
  Predicción      Recomendación     Chat LLM         Tracking
  Individual      Top N productos   Groq API         Experimentos
```

## Troubleshooting

**Airflow no inicia**: Verificar logs con `docker compose logs airflow-scheduler`

**DAG no aparece**: Verificar archivo en `dags/pipeline.py` y ejecutar `docker compose exec airflow-scheduler airflow dags list`

**Error de permisos**: `chmod -R 777 data/ logs/ mlruns/`

**Reiniciar desde cero**:
```bash
docker compose down -v
rm -rf data/models/* data/predictions/* logs/* mlruns/*
docker compose up -d
```