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
- **Airflow**: http://localhost:8080 (airflow/airflow)
- **MLflow**: http://localhost:5001

### Paso 4: Ejecutar DAG
Desde la interfaz web de Airflow:
1. Activar toggle del DAG `sodai_ml_pipeline`
2. Click en "Trigger DAG" (▶️)
3. Monitorear en vista "Graph"

O desde terminal:
```bash
docker compose exec airflow-scheduler airflow dags trigger sodai_ml_pipeline
```

### Verificar resultados
```bash
ls data/models/        # modelo_*.pkl, pipeline_pp.pkl, shap_summary.png
ls data/predictions/   # predicciones_*.csv
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