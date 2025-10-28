# Pipeline de Airflow - SodAI Drinks 🥤

## Descripción General del DAG

Este DAG implementa un pipeline completo de MLOps que automatiza el ciclo de vida del modelo predictivo, desde la ingesta de datos hasta la generación de predicciones. El pipeline está diseñado para operar en producción, detectando automáticamente drift en los datos y reentrenando el modelo cuando sea necesario.

### Características Principales

- **Automatización completa**: Ingesta → Preprocesamiento → Detección de Drift → Reentrenamiento Condicional → Predicción
- **Gestión inteligente de modelos**: Verifica existencia de modelo antes de drift detection
- **Detección de drift**: Sistema automático usando test de Kolmogorov-Smirnov
- **Reentrenamiento condicional**: Optuna (primera vez) o parámetros fijos (reentrenamiento)
- **Tracking con MLflow**: Registro de métricas, hiperparámetros, modelos y artefactos SHAP
- **Optimización con Optuna**: 30 trials para encontrar mejores hiperparámetros en primer entrenamiento
- **Interpretabilidad**: Generación de gráficos SHAP para cada entrenamiento
- **Predicciones semanales**: Generación automática de predicciones para la siguiente semana con preprocesamiento completo

---

## Descripción de Tareas del DAG

El DAG `sodai_ml_pipeline` está compuesto por **7 tareas**, organizadas en un flujo lógico con branching condicional:

### 1. **ingestar_datos**
- **Función**: Carga los datos de transacciones, productos y clientes desde archivos Parquet
- **Entrada**: 
  - `/opt/airflow/data/raw/transacciones.parquet`
  - `/opt/airflow/data/raw/productos.parquet`
  - `/opt/airflow/data/raw/clientes.parquet`
- **Salida**: DataFrames pusheados a XCom para las siguientes tareas
- **Código**:
```python
def task_ingestar_datos(**context):
    transacciones = pd.read_parquet('/opt/airflow/data/raw/transacciones.parquet')
    productos = pd.read_parquet('/opt/airflow/data/raw/productos.parquet')
    clientes = pd.read_parquet('/opt/airflow/data/raw/clientes.parquet')
    
    context['ti'].xcom_push(key='transacciones', value=transacciones.to_json())
    context['ti'].xcom_push(key='productos', value=productos.to_json())
    context['ti'].xcom_push(key='clientes', value=clientes.to_json())
```

### 2. **preprocesar_datos**
- **Función**: Aplica el pipeline de preprocesamiento completo de la Entrega 1 (14 pasos)
- **Pipeline**:
  1. Merge de transacciones con productos y clientes
  2. Filtro de transacciones con precio > 0
  3. Creación de variables temporales (mes, trimestre, semana)
  4. Agregaciones de frecuencia por cliente-producto
  5. Agregaciones de frecuencia por marca
  6. Agregaciones de frecuencia por categoría
  7. Geo-clustering con KMeans (4 clusters en latitud/longitud)
  8. Selección de features finales
  9. Eliminación de outliers con IQR (reemplaza con NaN, NO filtra filas)
  10. Split temporal (70% train, 15% val, 15% test)
  11. Adaptive undersampling (ratio mínimo entre 10:1 y máximo posible)
  12. One-Hot Encoding de variables categóricas
  13. Estandarización de variables numéricas
  14. Alineación de columnas entre conjuntos
- **Salida**: 
  - Arrays numpy guardados en `/tmp/` para train/val/test
  - Pipeline de preprocesamiento serializado: `/opt/airflow/data/models/pipeline_pp.pkl`
- **Dependencia**: `ingestar_datos`

### 3. **detectar_drift**
- **Función**: Verifica existencia de modelo y detecta drift en distribuciones de features
- **Lógica de decisión**:
```python
# 1. Primero verifica si existe un modelo previo
result = subprocess.run(['ls', '/opt/airflow/data/models/*.pkl'], ...)
if result.returncode != 0:
    # No hay modelo → Directamente ir a reentrenar
    return 'reentrenar_modelo'

# 2. Si existe modelo, detecta drift usando KS-test
for col in X_prod.columns:
    statistic, p_value = ks_2samp(X_train[col], X_prod[col])
    if p_value < 0.05:
        drift_detected = True
        
if drift_detected:
    return 'reentrenar_modelo'
else:
    return 'usar_modelo_existente'
```
- **Método**: Test de Kolmogorov-Smirnov (umbral p-value < 0.05)
- **Salida**: Branch decision (reentrenar_modelo o usar_modelo_existente)
- **Dependencia**: `preprocesar_datos`

### 4a. **reentrenar_modelo** (Branch condicional)
- **Función**: Entrena DecisionTreeClassifier con optimización condicional
- **Lógica de optimización**:
```python
# Verifica si hay modelos existentes
result = subprocess.run(['ls', '/opt/airflow/data/models/*.pkl'], ...)
if result.returncode != 0:
    # No hay modelos → Usar Optuna
    optimize = True
else:
    # Ya hay modelos → Usar parámetros fijos
    optimize = False

model, metrics = train_and_log_model(X_train, X_val, y_train, y_val, optimize=optimize)
```
- **Optimización con Optuna** (si `optimize=True`):
  - 30 trials de búsqueda bayesiana
  - Espacio de búsqueda:
    * `max_depth`: 5-20
    * `min_samples_split`: 2-20
    * `min_samples_leaf`: 1-10
  - Métrica objetivo: F1-score macro
  - Mejor resultado encontrado: `max_depth=16, min_samples_split=2, min_samples_leaf=10` (F1=0.5136)
- **Parámetros fijos** (si `optimize=False`):
  - Parámetros de Entrega 1: `max_depth=14, min_samples_split=10, min_samples_leaf=8`
  - Clustering: `n_clusters=4`
- **Tracking MLflow**:
  - Parámetros del modelo
  - Métricas por clase: precision, recall, f1-score
  - Métricas macro: accuracy, precision_macro, recall_macro, f1_macro
  - Artefactos: modelo .pkl, gráfico SHAP
- **Salida**: Modelo guardado en `/opt/airflow/data/models/modelo_YYYYMMDD_HHMMSS.pkl`
- **Dependencia**: `detectar_drift` (si drift=True o no hay modelo)

### 4b. **usar_modelo_existente** (Branch condicional)
- **Función**: Carga el modelo existente más reciente si no hay drift
- **Lógica**:
```python
# Busca el modelo más reciente en el directorio
archivos_modelo = sorted(glob.glob('/opt/airflow/data/models/modelo_*.pkl'))
if archivos_modelo:
    modelo_path = archivos_modelo[-1]
    model = joblib.load(modelo_path)
```
- **Salida**: Modelo en memoria pusheado a XCom
- **Dependencia**: `detectar_drift` (si drift=False y modelo existe)

### 5. **generar_predicciones**
- **Función**: Genera predicciones para la semana siguiente con preprocesamiento completo
- **Proceso**:
```python
# 1. Cargar modelo y pipeline de preprocesamiento
pipeline_pp = joblib.load('/opt/airflow/data/models/pipeline_pp.pkl')

# 2. Identificar semana actual y crear combinaciones para semana siguiente
max_semana = transacciones['semana'].max()
semana_siguiente = max_semana + 1

# 3. Crear todas las combinaciones cliente-producto para esa semana
combinaciones = pd.DataFrame({
    'cliente_id': todos_clientes,
    'producto_id': todos_productos,
    'semana': semana_siguiente
})

# 4. Aplicar preprocesamiento completo (merge, features, transform)
X_pred_transformed = pipeline_pp.transform(combinaciones)

# 5. Generar probabilidades
probabilidades = model.predict_proba(X_pred_transformed)[:, 1]

# 6. Guardar predicciones
predictions_df = pd.DataFrame({
    'cliente_id': combinaciones['cliente_id'],
    'producto_id': combinaciones['producto_id'],
    'semana': semana_siguiente,
    'probabilidad_compra': probabilidades
})
predictions_df.to_csv(f'/opt/airflow/data/predictions/predicciones_{timestamp}.csv')
```
- **Salida**: CSV con 5000 predicciones (formato: cliente_id, producto_id, semana, probabilidad_compra)
- **Trigger rule**: `one_success` (se ejecuta si cualquiera de las ramas anteriores tuvo éxito)
- **Dependencia**: `reentrenar_modelo` o `usar_modelo_existente`

### 6. **fin_pipeline**
- **Función**: Tarea de cierre y consolidación
- **Propósito**: Marca el fin exitoso del pipeline
- **Dependencia**: `generar_predicciones`

---

## Diagrama de Flujo del Pipeline

```
                    ┌─────────────────────┐
                    │  ingestar_datos     │
                    │                     │
                    │ • Carga parquets    │
                    │ • Push a XCom       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ preprocesar_datos   │
                    │                     │
                    │ • 14 pasos          │
                    │ • Pipeline sklearn  │
                    │ • Guarda arrays     │
                    │ • Guarda pipeline   │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────┐
              │    detectar_drift          │
              │  (BranchPythonOperator)    │
              │                            │
              │ 1. ¿Existe modelo?         │
              │    NO → reentrenar         │
              │    SÍ → continuar          │
              │                            │
              │ 2. ¿Hay drift (KS-test)?   │
              │    SÍ → reentrenar         │
              │    NO → usar existente     │
              └────┬───────────────┬───────┘
                   │               │
       (No modelo  │               │  (Modelo OK
       o drift)    │               │   + no drift)
                   │               │
                   ▼               ▼
         ┌───────────────┐  ┌──────────────────┐
         │ reentrenar_   │  │ usar_modelo_     │
         │ modelo        │  │ existente        │
         │               │  │                  │
         │ ¿Optimizar?   │  │ • Carga último   │
         │ • Sin modelos │  │   .pkl           │
         │   → Optuna    │  │                  │
         │   (30 trials) │  └──────┬───────────┘
         │ • Con modelos │          │
         │   → Params    │          │
         │   fijos       │          │
         │               │          │
         │ • MLflow log  │          │
         │ • SHAP plots  │          │
         └───────┬───────┘          │
                 │                  │
                 └────────┬─────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ generar_predicciones │
                │  (trigger_rule:      │
                │   one_success)       │
                │                      │
                │ • Carga pipeline     │
                │ • Semana siguiente   │
                │ • Transform          │
                │ • Predict_proba      │
                │ • Guarda CSV         │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    fin_pipeline      │
                └──────────────────────┘
```

### Notas sobre el flujo:
- **Branching condicional**: La tarea `detectar_drift` usa `BranchPythonOperator` para decidir el camino
- **Optimización inteligente**: Solo se ejecuta Optuna en el primer entrenamiento, luego usa parámetros fijos
- **Trigger rule especial**: `generar_predicciones` tiene `trigger_rule='one_success'` para ejecutarse si cualquiera de las ramas anteriores tuvo éxito
- **Persistencia**: El pipeline de preprocesamiento se guarda y reutiliza en predicciones

---

## Representación Visual del DAG en Airflow

### Vista del DAG en la Interfaz de Airflow

Para acceder a la interfaz de Airflow:

1. Asegúrate de que los contenedores estén corriendo:
```bash
docker compose ps
```

2. Accede a la interfaz web:
```
URL: http://localhost:8080
Usuario: admin
Password: admin
```

3. En la vista Graph del DAG `sodai_ml_pipeline` podrás ver:

**Características visibles en la interfaz:**
- ✅ **Color verde**: Tareas completadas exitosamente
- 🟡 **Color amarillo**: Tareas en ejecución
- 🔴 **Color rojo**: Tareas fallidas
- 🔵 **Color azul claro**: Tareas omitidas (skipped) debido al branching
- **Líneas de conexión**: Muestran el flujo de dependencias entre tareas
- **Ramificación**: El `BranchPythonOperator` en `detectar_drift` crea dos caminos posibles

**Ejemplo de captura de pantalla que deberías ver:**

![Vista Graph del DAG](./screenshots/airflow_dag_graph.png)

*Nota: Para capturar esta imagen, ve a DAGs → sodai_ml_pipeline → Graph y toma un screenshot de la vista completa del pipeline.*

### Vista de Grid

La vista Grid muestra el historial de ejecuciones:

![Vista Grid del DAG](./screenshots/airflow_dag_grid.png)

Aquí puedes ver:
- Historial de todas las ejecuciones
- Estado de cada tarea en cada run
- Duración de las tareas
- Cuándo se ejecutó cada rama del branching

### Logs de Ejecución

Para ver los logs de una tarea específica:
1. Click en la tarea en el Graph
2. Click en "Log"
3. Verás el output detallado incluyendo:
   - Mensajes de logging personalizados
   - Resultados de Optuna (si se ejecutó)
   - Métricas de MLflow
   - Errores si los hubo

---

## Detección de Drift y Lógica de Reentrenamiento

### 1. Sistema de Detección de Drift

El pipeline implementa un sistema robusto de detección de drift que primero verifica la existencia de un modelo antes de realizar la comparación estadística.

#### Paso 1: Verificación de Existencia de Modelo

```python
def task_detectar_drift(**context):
    # PRIMERO: Verificar si existe un modelo previo
    result = subprocess.run(
        ['ls', '/opt/airflow/data/models/*.pkl'],
        capture_output=True,
        text=True,
        shell=True
    )
    
    if result.returncode != 0:
        logging.info("No se encontró modelo previo. Iniciando entrenamiento...")
        return 'reentrenar_modelo'
    
    logging.info("Modelo previo encontrado. Procediendo a detectar drift...")
    # Continuar con detección de drift
```

**Justificación**: 
- En la primera ejecución del pipeline no existe un modelo previo
- No tiene sentido detectar drift si no hay un modelo de referencia
- Esta verificación evita errores y toma la decisión correcta (entrenar)

#### Paso 2: Detección de Drift con Test de Kolmogorov-Smirnov

Si existe un modelo previo, se procede a comparar distribuciones:

```python
from scipy.stats import ks_2samp

# Cargar datos de entrenamiento (referencia) y producción (nuevos)
X_train = np.load('/tmp/X_train.npy', allow_pickle=True)
X_prod = np.load('/tmp/X_test.npy', allow_pickle=True)  # Simula datos nuevos

# Convertir a DataFrames para análisis
X_train_df = pd.DataFrame(X_train)
X_prod_df = pd.DataFrame(X_prod)

# Detectar drift en cada feature
drift_detected = False
for col in X_train_df.columns:
    statistic, p_value = ks_2samp(X_train_df[col], X_prod_df[col])
    
    if p_value < 0.05:  # Nivel de significancia del 5%
        logging.warning(f"Drift detectado en columna {col}: p-value={p_value:.4f}")
        drift_detected = True
    else:
        logging.info(f"No drift en columna {col}: p-value={p_value:.4f}")

# Decisión de branching
if drift_detected:
    logging.info("Se detectó drift. Reentrenando modelo...")
    return 'reentrenar_modelo'
else:
    logging.info("No se detectó drift. Usando modelo existente.")
    return 'usar_modelo_existente'
```

**Explicación del método**:
- **Test de Kolmogorov-Smirnov**: Prueba estadística no paramétrica que compara dos distribuciones
- **Hipótesis nula**: Las dos distribuciones son iguales
- **P-value < 0.05**: Rechazamos la hipótesis nula → hay diferencia significativa (drift)
- **Ventajas**:
  - No asume distribución normal
  - Detecta cambios en forma, localización y escala
  - Interpretable: p-value bajo = distribuciones diferentes

**Configuración de sensibilidad**:
```python
DRIFT_THRESHOLD = 0.05  # Nivel de significancia
# Valores más bajos (ej. 0.01) → menos sensible, detecta solo cambios grandes
# Valores más altos (ej. 0.10) → más sensible, detecta cambios pequeños
```

### 2. Lógica de Reentrenamiento Condicional

El pipeline implementa una estrategia inteligente que diferencia entre el **primer entrenamiento** y **reentrenamientos posteriores**.

#### Estrategia 1: Primer Entrenamiento (con Optuna)

Cuando no existen modelos previos:

```python
def task_reentrenar(**context):
    # Verificar si es el primer entrenamiento
    result = subprocess.run(
        ['ls', '/opt/airflow/data/models/*.pkl'],
        capture_output=True,
        text=True,
        shell=True
    )
    
    if result.returncode != 0:
        # NO HAY MODELOS → Primera vez → Usar Optuna
        logging.info("Primer entrenamiento: optimizando hiperparámetros con Optuna...")
        optimize = True
    else:
        # YA HAY MODELOS → Reentrenamiento → Usar params fijos
        logging.info("Reentrenamiento: usando hiperparámetros fijos...")
        optimize = False
    
    # Entrenar con la estrategia correspondiente
    model, metrics = train_and_log_model(
        X_train, X_val, y_train, y_val, 
        optimize=optimize
    )
```

**Justificación**:
- **Primera vez**: No sabemos qué hiperparámetros funcionan mejor
  - Usamos Optuna para explorar el espacio de búsqueda
  - 30 trials de optimización bayesiana
  - Métrica objetivo: F1-score macro
  
- **Reentrenamientos**: Ya tenemos hiperparámetros validados
  - Usar Optuna cada vez sería costoso computacionalmente
  - Los hiperparámetros óptimos tienden a ser estables
  - Usamos los parámetros encontrados en Entrega 1 (validados)

#### Estrategia 2: Optimización con Optuna (Primer Entrenamiento)

```python
def optimize_with_optuna(X_train, X_val, y_train, y_val):
    """
    Optimiza hiperparámetros usando Optuna
    Returns: dict con mejores hiperparámetros
    """
    def objective(trial):
        # Espacio de búsqueda
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
        
        # Entrenar modelo con estos parámetros
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        return f1  # Maximizar F1-score
    
    # Crear estudio de optimización
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    logging.info(f"Mejores hiperparámetros encontrados: {study.best_params}")
    logging.info(f"Mejor F1-score: {study.best_value:.4f}")
    
    return study.best_params
```

**Resultado de Optuna en ejecución real**:
```
Mejores hiperparámetros:
- max_depth: 16
- min_samples_split: 2
- min_samples_leaf: 10
- F1-score: 0.5136
```

#### Estrategia 3: Parámetros Fijos (Reentrenamientos)

```python
# Parámetros validados de Entrega 1
FIXED_PARAMS = {
    'max_depth': 14,
    'min_samples_split': 10,
    'min_samples_leaf': 8,
    'random_state': 42
}

# Para geo-clustering
CLUSTERING_PARAMS = {
    'n_clusters': 4
}
```

**Justificación de parámetros fijos**:
- Ya fueron validados en Entrega 1 con el mismo dataset
- Proporcionan buen balance entre sesgo y varianza
- Evitan overfitting con `min_samples_leaf=8`
- Son estables ante pequeños cambios en los datos

### 3. Registro en MLflow

Ambas estrategias registran información completa en MLflow:

```python
def train_and_log_model(X_train, X_val, y_train, y_val, optimize=False):
    try:
        mlflow.set_tracking_uri("http://mlflow:5001")
        mlflow.set_experiment("sodai_production")
        
        with mlflow.start_run():
            # Obtener hiperparámetros
            if optimize:
                logging.info("Optimizando con Optuna...")
                best_params = optimize_with_optuna(X_train, X_val, y_train, y_val)
                mlflow.log_param("optimization_method", "optuna")
                mlflow.log_param("n_trials", 30)
            else:
                logging.info("Usando parámetros fijos...")
                best_params = FIXED_PARAMS
                mlflow.log_param("optimization_method", "fixed")
            
            # Entrenar modelo
            model = DecisionTreeClassifier(**best_params)
            model.fit(X_train, y_train)
            
            # Evaluar y loguear métricas
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Métricas generales
            mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred_train))
            mlflow.log_metric("val_accuracy", accuracy_score(y_val, y_pred_val))
            
            # Métricas por clase
            report = classification_report(y_val, y_pred_val, output_dict=True)
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metric(f"{label}_precision", metrics['precision'])
                    mlflow.log_metric(f"{label}_recall", metrics['recall'])
                    mlflow.log_metric(f"{label}_f1", metrics['f1-score'])
            
            # Métricas macro
            mlflow.log_metric("val_precision_macro", report['macro avg']['precision'])
            mlflow.log_metric("val_recall_macro", report['macro avg']['recall'])
            mlflow.log_metric("val_f1_macro", report['macro avg']['f1-score'])
            
            # Loguear hiperparámetros
            for param, value in best_params.items():
                mlflow.log_param(param, value)
            
            # Generar y loguear SHAP plots
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_val[:100])
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_val[:100], show=False)
                plt.savefig('/opt/airflow/data/models/shap_summary.png')
                plt.close()
                
                mlflow.log_artifact('/opt/airflow/data/models/shap_summary.png')
                logging.info("SHAP plot generado y registrado en MLflow")
            except Exception as e:
                logging.warning(f"No se pudo generar SHAP plot: {e}")
            
    except Exception as e:
        logging.error(f"Error en MLflow tracking: {e}")
        logging.info("Continuando sin tracking de MLflow...")
    
    return model, report
```

### 4. Flujo Completo de Decisión

```
┌──────────────────────────────────────┐
│ ¿Existe modelo previo en /models/?  │
└─────────┬─────────────┬──────────────┘
          │ NO          │ SÍ
          ▼             ▼
    ┌─────────┐   ┌────────────────┐
    │Entrenar │   │ Detectar Drift │
    │(Optuna) │   │   (KS-test)    │
    └─────────┘   └────┬──────┬────┘
                       │ SÍ   │ NO
                       ▼      ▼
                  ┌────────┐ ┌──────────┐
                  │Entrenar│ │  Usar    │
                  │(Fixed) │ │ Existente│
                  └────────┘ └──────────┘
```

### 5. Ventajas del Diseño Implementado

✅ **Eficiencia**: Optuna solo en primera ejecución (costoso computacionalmente)

✅ **Robustez**: Detección estadística de drift con método validado

✅ **Flexibilidad**: Umbral de drift configurable según necesidades del negocio

✅ **Trazabilidad**: Todo registrado en MLflow (parámetros, métricas, estrategia)

✅ **Interpretabilidad**: SHAP plots para cada modelo entrenado

✅ **Producción-ready**: Maneja casos edge (sin modelo, con drift, sin drift)

### 6. Acceso a MLflow para Ver Experimentos

Para revisar los experimentos y métricas:

```bash
# Asegúrate de que MLflow esté corriendo
docker compose ps mlflow

# Accede a la interfaz
URL: http://localhost:5001
```

En la interfaz podrás ver:
- Todos los runs del experimento `sodai_production`
- Comparación de métricas entre runs
- Hiperparámetros utilizados
- Artefactos (modelos .pkl, gráficos SHAP)
- Método de optimización usado (Optuna vs Fixed)

---

## Configuración del Entorno

### Arquitectura de Contenedores

El sistema utiliza Docker Compose con 4 servicios:

```yaml
services:
  postgres:         # Base de datos para Airflow
  airflow-init:     # Inicialización de Airflow (una vez)
  airflow-scheduler: # Orquestador de tareas
  mlflow:           # Servidor de tracking de experimentos
```

### Requisitos del Sistema

**Software necesario:**
- Docker Desktop 4.0+ (macOS/Windows) o Docker Engine 20.10+ (Linux)
- Docker Compose 2.0+
- 8GB RAM mínimo (recomendado 16GB)
- 10GB espacio en disco

### Dependencias de Python

```txt
# requirements.txt
apache-airflow==2.7.3
mlflow==2.8.0
optuna==3.4.0
pandas==2.1.3
numpy==1.24.0
scikit-learn==1.3.2
scipy==1.11.4
shap==0.43.0
pyarrow==14.0.1
matplotlib==3.8.2
joblib==1.3.2
```

### Estructura de Directorios

```
airflow/
├── dags/
│   ├── pipeline.py              # Definición del DAG
│   └── helper_functions.py      # Funciones de ML y transformers
├── data/
│   ├── raw/                     # Datos de entrada (parquets)
│   ├── models/                  # Modelos entrenados (.pkl)
│   └── predictions/             # Predicciones generadas (.csv)
├── logs/                        # Logs de Airflow
│   ├── dag_processor_manager/
│   └── scheduler/
├── mlruns/                      # Experimentos de MLflow
├── docker-compose.yml           # Orquestación de contenedores
├── Dockerfile                   # Imagen personalizada de Airflow
├── requirements.txt             # Dependencias de Python
└── README.md                    # Esta documentación
```

### Configuración de Docker Compose

**Variables de entorno clave:**

```yaml
environment:
  # Airflow
  AIRFLOW__CORE__EXECUTOR: LocalExecutor
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
  AIRFLOW__CORE__FERNET_KEY: ''
  AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
  AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
  
  # MLflow
  MLFLOW_TRACKING_URI: http://mlflow:5001
```

**Puertos expuestos:**
- `8080`: Interfaz web de Airflow
- `5001`: Interfaz web de MLflow

**Volúmenes montados:**
```yaml
volumes:
  - ./dags:/opt/airflow/dags
  - ./data:/opt/airflow/data
  - ./logs:/opt/airflow/logs
  - ./mlruns:/mlflow/mlruns
```

### Configuración de MLflow

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.8.0
  ports:
    - "5001:5001"
  command: >
    mlflow server
    --host 0.0.0.0
    --port 5001
    --backend-store-uri file:///mlflow/mlruns
    --default-artifact-root /mlflow/mlartifacts
  volumes:
    - ./mlruns:/mlflow/mlruns
    - ./mlartifacts:/mlflow/mlartifacts
```

**Nota**: Se usa almacenamiento basado en archivos (`file://`) en lugar de SQLite para evitar problemas de permisos.

---

## Ejecución del Pipeline

### 1. Configuración Inicial

**Paso 1: Clonar y navegar al directorio**
```bash
cd /path/to/Proyecto/entrega2/airflow
```

**Paso 2: Verificar estructura de datos**
```bash
ls -lh data/raw/
# Deberías ver:
# - transacciones.parquet
# - productos.parquet
# - clientes.parquet
```

**Paso 3: Levantar servicios con Docker Compose**
```bash
docker compose up -d
```

**Paso 4: Verificar que los contenedores estén corriendo**
```bash
docker compose ps

# Salida esperada:
# NAME                    STATUS
# airflow-scheduler       Up
# postgres                Up (healthy)
# mlflow_server           Up
```

**Paso 5: Esperar inicialización de Airflow (~30 segundos)**
```bash
# Verificar logs del scheduler
docker compose logs -f airflow-scheduler

# Buscar mensaje:
# "Loaded executor: LocalExecutor"
# "DAG sodai_ml_pipeline loaded"
```

### 2. Acceso a las Interfaces

**Airflow Web UI:**
```
URL: http://localhost:8080
Usuario: admin
Password: admin
```

**MLflow Web UI:**
```
URL: http://localhost:5001
# No requiere autenticación
```

### 3. Ejecución Manual del DAG

**Opción A: Desde la interfaz web**

1. Ve a http://localhost:8080
2. Login con admin/admin
3. Busca el DAG `sodai_ml_pipeline`
4. Activa el toggle (cambiar de Paused a Active)
5. Click en "Trigger DAG" (botón de play ▶️)
6. Ve a la vista "Graph" para monitorear progreso

**Opción B: Desde la línea de comandos**

```bash
# Activar el DAG
docker compose exec airflow-scheduler airflow dags unpause sodai_ml_pipeline

# Trigger manual
docker compose exec airflow-scheduler airflow dags trigger sodai_ml_pipeline

# Ver estado
docker compose exec airflow-scheduler airflow dags list-runs -d sodai_ml_pipeline
```

### 4. Ejecución Programada

Para configurar ejecución automática periódica, modifica el DAG:

```python
# En dags/pipeline.py
dag = DAG(
    'sodai_ml_pipeline',
    default_args=default_args,
    description='Pipeline ML completo con drift detection',
    schedule_interval='0 2 * * 1',  # Todos los lunes a las 2 AM
    catchup=False,
    max_active_runs=1,
    start_date=datetime(2024, 1, 1),
)
```

**Opciones de schedule_interval:**
- `'@daily'`: Todos los días a medianoche
- `'@weekly'`: Todos los domingos a medianoche
- `'0 2 * * 1'`: Todos los lunes a las 2 AM
- `'0 */6 * * *'`: Cada 6 horas
- `None`: Solo ejecución manual

### 5. Monitoreo de Ejecución

**Ver logs de una tarea específica:**

1. En la interfaz web: DAGs → sodai_ml_pipeline → Graph
2. Click en la tarea que quieres revisar
3. Click en "Log"

**Logs desde terminal:**

```bash
# Logs del scheduler (orquestación)
docker compose logs -f airflow-scheduler

# Logs de una tarea específica
docker compose exec airflow-scheduler airflow tasks logs sodai_ml_pipeline ingestar_datos <run_id>
```

**Métricas en MLflow:**

1. Ve a http://localhost:5001
2. Click en el experimento `sodai_production`
3. Revisa:
   - Parámetros (max_depth, min_samples_split, etc.)
   - Métricas (accuracy, f1_macro, etc.)
   - Artefactos (shap_summary.png)

### 6. Verificar Resultados

**Modelo entrenado:**
```bash
ls -lh data/models/
# Deberías ver:
# - modelo_YYYYMMDD_HHMMSS.pkl
# - pipeline_pp.pkl
# - shap_summary.png
```

**Predicciones generadas:**
```bash
ls -lh data/predictions/
# Deberías ver:
# - predicciones_YYYYMMDD_HHMMSS.csv

# Ver preview
head data/predictions/predicciones_*.csv
```

**Estructura del archivo de predicciones:**
```csv
cliente_id,producto_id,semana,probabilidad_compra
1,101,53,0.234
1,102,53,0.789
2,101,53,0.123
...
```

---

## Reproducibilidad

### Pasos Completos para Reproducir desde Cero

**1. Preparación del entorno**
```bash
# Clonar repositorio (o descargar archivos)
cd /path/to/Proyecto/entrega2/airflow

# Verificar que tengas Docker instalado
docker --version
docker compose version
```

**2. Preparar datos de entrada**
```bash
# Asegúrate de tener los archivos parquet en data/raw/
ls data/raw/
# transacciones.parquet
# productos.parquet
# clientes.parquet
```

**3. Levantar infraestructura**
```bash
# Construir imagen personalizada (si hay cambios)
docker compose build

# Iniciar servicios en background
docker compose up -d

# Verificar salud de los contenedores
docker compose ps
```

**4. Primera ejecución (sin modelo previo)**
```bash
# Asegúrate de que el directorio de modelos esté vacío
rm -f data/models/*.pkl

# Trigger DAG desde terminal
docker compose exec airflow-scheduler airflow dags trigger sodai_ml_pipeline

# O desde la interfaz web (http://localhost:8080)
```

**Resultado esperado:**
```
ingestar_datos → SUCCESS
preprocesar_datos → SUCCESS
detectar_drift → DECISION (no hay modelo previo)
reentrenar_modelo → SUCCESS (con Optuna, 30 trials)
generar_predicciones → SUCCESS
fin_pipeline → SUCCESS

Artefactos generados:
- modelo_20251027_XXXXXX.pkl
- pipeline_pp.pkl
- shap_summary.png
- predicciones_20251027_XXXXXX.csv
```

**5. Segunda ejecución (con modelo existente, sin drift)**
```bash
# Trigger nuevamente
docker compose exec airflow-scheduler airflow dags trigger sodai_ml_pipeline
```

**Resultado esperado:**
```
ingestar_datos → SUCCESS
preprocesar_datos → SUCCESS
detectar_drift → DECISION (modelo existe, no drift)
usar_modelo_existente → SUCCESS
generar_predicciones → SUCCESS
fin_pipeline → SUCCESS

Nota: NO se ejecutó reentrenar_modelo (skipped)
```

**6. Tercera ejecución (forzar reentrenamiento sin Optuna)**

Para simular reentrenamiento por drift (usando parámetros fijos):

```bash
# Editar helper_functions.py temporalmente para forzar drift
# O esperar a tener datos reales con drift

# Trigger DAG
docker compose exec airflow-scheduler airflow dags trigger sodai_ml_pipeline
```

**Resultado esperado:**
```
detectar_drift → DECISION (drift detectado)
reentrenar_modelo → SUCCESS (con parámetros fijos, NO Optuna)

Modelo entrenado con:
- max_depth: 14
- min_samples_split: 10
- min_samples_leaf: 8
```

### Limpieza y Reinicio

**Detener servicios:**
```bash
docker compose down
```

**Reiniciar desde cero (borra todo):**
```bash
# Detener y eliminar contenedores, volúmenes, redes
docker compose down -v

# Limpiar datos generados
rm -rf data/models/*
rm -rf data/predictions/*
rm -rf logs/*
rm -rf mlruns/*

# Iniciar nuevamente
docker compose up -d
```

**Mantener datos pero reiniciar contenedores:**
```bash
docker compose restart
```

### Troubleshooting Común

**Problema: Airflow no inicia**
```bash
# Verificar logs
docker compose logs airflow-scheduler

# Solución común: Reiniciar servicios
docker compose restart
```

**Problema: DAG no aparece en la interfaz**
```bash
# Verificar que el archivo esté en el lugar correcto
ls dags/pipeline.py

# Forzar reescaneo de DAGs
docker compose exec airflow-scheduler airflow dags list
```

**Problema: MLflow no accesible**
```bash
# Verificar que el contenedor esté corriendo
docker compose ps mlflow

# Revisar logs
docker compose logs mlflow

# Reiniciar solo MLflow
docker compose restart mlflow
```

**Problema: Error de permisos en archivos**
```bash
# Dar permisos a directorios montados
chmod -R 777 data/ logs/ mlruns/
```

**Problema: Memoria insuficiente**
```bash
# Aumentar memoria asignada a Docker Desktop
# Settings → Resources → Memory → 8GB mínimo
```

---

## Estructura de Archivos Completa

```
airflow/
│
├── dags/
│   ├── pipeline.py                    # DAG principal (7 tareas)
│   └── helper_functions.py            # Transformers + ML functions
│
├── data/
│   ├── raw/                           # Input data (no modificar)
│   │   ├── transacciones.parquet
│   │   ├── productos.parquet
│   │   └── clientes.parquet
│   │
│   ├── models/                        # Modelos y artefactos generados
│   │   ├── modelo_YYYYMMDD_HHMMSS.pkl
│   │   ├── pipeline_pp.pkl
│   │   └── shap_summary.png
│   │
│   └── predictions/                   # Predicciones generadas
│       └── predicciones_YYYYMMDD_HHMMSS.csv
│
├── logs/                              # Logs de Airflow (auto-generado)
│   ├── dag_processor_manager/
│   └── scheduler/
│       └── 2024-10-27/
│
├── mlruns/                            # Experimentos de MLflow
│   └── 0/                             # Experiment ID
│       └── <run_id>/
│           ├── metrics/
│           ├── params/
│           └── artifacts/
│
├── mlartifacts/                       # Artefactos de MLflow
│
├── docker-compose.yml                 # Definición de servicios
├── Dockerfile                         # Imagen custom de Airflow
├── requirements.txt                   # Dependencias Python
└── README.md                          # Este archivo
```

---

## Contacto y Soporte

**Equipo de Desarrollo:**
- Proyecto: SodAI Drinks - Entrega 2
- Curso: Laboratorio MDS 2025-2
- Universidad: [Tu Universidad]

**Recursos Útiles:**
- Documentación de Airflow: https://airflow.apache.org/docs/
- Documentación de MLflow: https://mlflow.org/docs/latest/
- Documentación de Optuna: https://optuna.readthedocs.io/

**Reportar Issues:**
Si encuentras problemas con el pipeline, revisa:
1. Logs de Airflow (interfaz web o terminal)
2. Logs de MLflow
3. Estado de los contenedores (`docker compose ps`)
4. Permisos de archivos (`ls -la data/`)

---

## Versiones y Changelog

**v1.0.0 (2024-10-27)**
- ✅ Pipeline completo de 7 tareas implementado
- ✅ Detección de drift con KS-test
- ✅ Optimización condicional con Optuna (30 trials)
- ✅ Reentrenamiento inteligente (Optuna vs parámetros fijos)
- ✅ Tracking completo en MLflow
- ✅ SHAP plots para interpretabilidad
- ✅ Predicciones para semana siguiente con preprocesamiento completo
- ✅ Docker Compose para orquestación
- ✅ Documentación completa

---

*Este README fue generado como parte de la Entrega 2 del curso de Laboratorio MDS, cumpliendo con el requisito de documentación del pipeline de Airflow (0.5 puntos).*