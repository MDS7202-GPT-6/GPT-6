# API de Predicciones - SodAI Drinks

## Descripción

API REST construida con FastAPI que expone el modelo de predicción de compra semanal entrenado por el pipeline de Airflow. La aplicación permite realizar predicciones individuales, en lote y generar recomendaciones personalizadas de productos para clientes.

## Arquitectura

```
┌─────────────────────────────────────────────────┐
│              Airflow Pipeline                   │
│  (Entrena modelo y genera predicciones)         │
│                                                 │
│  Outputs:                                       │
│  • /data/models/modelo_*.pkl                   │
│  • /data/predictions/predicciones_*.csv        │
│  • /data/raw/*.parquet (datos base)            │
└─────────────┬───────────────────────────────────┘
              │ (volúmenes compartidos)
              ▼
┌─────────────────────────────────────────────────┐
│               Docker App                        │
│                                                 │
│  ┌──────────────┐         ┌──────────────┐    │
│  │   Backend    │◄────────│   Frontend   │    │
│  │  (FastAPI)   │         │  (Gradio)    │    │
│  │  Port 8000   │         │  Port 7860   │    │
│  └──────────────┘         └──────────────┘    │
│         │                                       │
│         │ Carga modelo y datos                 │
│         ▼                                       │
│  /app/models/modelo_*.pkl                      │
│  /app/data/*.parquet                           │
└─────────────────────────────────────────────────┘
```

## Características

**Carga automática**: Detecta y carga el modelo más reciente de Airflow  
**Predicciones individuales**: Endpoint `/predict` para un cliente-producto  
**Recomendaciones personalizadas**: Endpoint `/recommend` que evalúa todos los productos para un cliente y devuelve top N  
**Predicciones en lote**: Endpoint `/predict_batch` para múltiples combinaciones  
**Acceso a predicciones históricas**: Endpoint `/latest_predictions` con predicciones de Airflow  
**Health checks**: Monitoreo del estado de la API  
**Documentación automática**: Swagger UI en `/docs`  
**Preprocesamiento integrado**: Aplica las mismas transformaciones que el pipeline de entrenamiento  
**Frontend Gradio**: Interfaz web amigable con 4 pestañas (Predicción, Recomendaciones, Estado del Sistema, Modelo)

## Requisitos Previos

### 1. Pipeline de Airflow ejecutado

Antes de levantar la app, asegúrate de que el pipeline de Airflow haya corrido al menos una vez:

```bash
cd ../airflow
docker compose up -d
# Esperar a que se genere el modelo en airflow/data/models/
```

Verifica que existan los archivos:
```bash
ls -lh ../airflow/data/models/modelo_*.pkl
ls -lh ../airflow/data/raw/*.parquet
```

### 2. Docker y Docker Compose

- Docker Desktop 4.0+ (macOS/Windows) o Docker Engine 20.10+ (Linux)
- Docker Compose 2.0+

## Instalación y Ejecución

### Paso 1: Navegar al directorio de la app

```bash
cd /path/to/Proyecto/entrega2/app
```

### Paso 2: Verificar estructura

```bash
ls -la
# Deberías ver:
# - backend/      (código del backend)
# - frontend/     (código del frontend)
# - docker-compose.yml
# - README.md
```

### Paso 3: Construir y levantar servicios

```bash
docker compose build
docker compose up -d
```

### Paso 4: Verificar que los contenedores estén corriendo

```bash
docker compose ps

# Salida esperada:
# NAME                STATUS              PORTS
# sodai_backend       Up (healthy)        0.0.0.0:8000->8000/tcp
# sodai_frontend      Up                  0.0.0.0:7860->7860/tcp
```

### Paso 5: Verificar logs

```bash
# Backend
docker compose logs backend

# Buscar estos mensajes:
# Modelo cargado desde: /app/models/modelo_20251027_000556.pkl
# Datos cargados: XXXXX transacciones
```

### Paso 6: Acceder a las interfaces

**Backend API (FastAPI):**
```
URL: http://localhost:8000
Documentación interactiva: http://localhost:8000/docs
Documentación alternativa: http://localhost:8000/redoc
```

**Frontend (Gradio):**
```
URL: http://localhost:7860
```

## Uso de la API

### 1. Health Check

Verificar que la API esté funcionando:

```bash
curl http://localhost:8000/health
```

**Respuesta esperada:**
```json
{
  "status": "healthy",
  "modelo_cargado": true,
  "datos_cargados": true,
  "timestamp": "2024-10-27T00:30:00"
}
```

### 2. Información del Modelo

Ver detalles del modelo cargado:

```bash
curl http://localhost:8000/model_info
```

**Respuesta esperada:**
```json
{
  "modelo_path": "/app/models/modelo_20251027_000556.pkl",
  "tipo_modelo": "DecisionTreeClassifier",
  "parametros": {
    "max_depth": 16,
    "min_samples_split": 2,
    "min_samples_leaf": 10,
    "random_state": 42
  },
  "n_features": 5
}
```

### 3. Predicción Individual

Predecir probabilidad de compra para un cliente-producto específico:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cliente_id": 1,
    "producto_id": 101,
    "semana": 53
  }'
```

**Respuesta esperada:**
```json
{
  "cliente_id": 1,
  "producto_id": 101,
  "semana": 53,
  "prediccion": 1,
  "probabilidad_compra": 0.8234,
  "timestamp": "2024-10-27T00:35:00"
}
```

**Campos:**
- `prediccion`: 0 (no comprará) o 1 (comprará)
- `probabilidad_compra`: Probabilidad entre 0 y 1
- `semana`: Si no se especifica, usa semana actual + 1

### 4. Predicción en Lote

Predecir para múltiples combinaciones:

```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "clientes": [1, 2, 3],
    "productos": [101, 102, 103],
    "semana": 53
  }'
```

**Respuesta esperada:**
```json
{
  "n_predicciones": 9,
  "semana": 53,
  "predicciones": [
    {
      "cliente_id": 1,
      "producto_id": 101,
      "semana": 53,
      "prediccion": 1,
      "probabilidad_compra": 0.8234
    },
    {
      "cliente_id": 1,
      "producto_id": 102,
      "semana": 53,
      "prediccion": 0,
      "probabilidad_compra": 0.2145
    },
    ...
  ],
  "timestamp": "2024-10-27T00:40:00"
}
```

**Nota**: Genera todas las combinaciones posibles (clientes × productos)

### 5. Recomendaciones de Productos

Generar top N productos recomendados para un cliente específico:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "cliente_id": 254403,
    "semana": 53,
    "top_n": 5
  }'
```

**Respuesta esperada:**
```json
{
  "cliente_id": 254403,
  "semana": 53,
  "año": 2024,
  "cliente_info": {
    "tipo": "RETAIL",
    "region": 1,
    "zona": 5
  },
  "n_productos_evaluados": 971,
  "top_n": 5,
  "recomendaciones": [
    {
      "ranking": 1,
      "producto_id": 33534,
      "probabilidad_compra": 0.899,
      "categoria": "BEBIDAS CARBONATADAS",
      "marca": "Brand 24",
      "sub_categoria": "GASEOSAS",
      "segmento": "LOW"
    },
    {
      "ranking": 2,
      "producto_id": 34567,
      "probabilidad_compra": 0.892,
      "categoria": "BEBIDAS CARBONATADAS",
      "marca": "Brand 24",
      "sub_categoria": "GASEOSAS",
      "segmento": "MEDIUM"
    }
  ],
  "timestamp": "2024-10-27T00:45:00"
}
```

**Campos:**
- `n_productos_evaluados`: Total de productos analizados (típicamente ~971)
- `recomendaciones`: Array con top N productos ordenados por probabilidad
- `ranking`: Posición en el ranking (1 = más probable)
- `semana`: Si no se especifica, usa semana actual + 1
- `top_n`: Número de recomendaciones (default 5, máximo 20)

**Casos de uso:**
- Campañas de marketing personalizadas
- Recomendaciones en app móvil
- Optimización de inventario por cliente

### 6. Últimas Predicciones de Airflow

Acceder a las predicciones generadas por el pipeline:

```bash
curl http://localhost:8000/latest_predictions
```

**Respuesta esperada:**
```json
{
  "archivo": "predicciones_20251027_000600.csv",
  "n_predicciones": 5000,
  "top_10_probabilidades": [
    {
      "cliente_id": 45,
      "producto_id": 203,
      "semana": 53,
      "probabilidad_compra": 0.9876
    },
    ...
  ]
}
```

## Uso desde Python

### Instalar requests

```bash
pip install requests
```

### Ejemplo de uso

```python
import requests
import pandas as pd

# URL base de la API
BASE_URL = "http://localhost:8000"

# 1. Verificar salud
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. Predicción individual
data = {
    "cliente_id": 1,
    "producto_id": 101,
    "semana": 53
}
response = requests.post(f"{BASE_URL}/predict", json=data)
result = response.json()
print(f"Probabilidad de compra: {result['probabilidad_compra']:.2%}")

# 3. Predicción en lote
data = {
    "clientes": [1, 2, 3, 4, 5],
    "productos": [101, 102, 103],
    "semana": 53
}
response = requests.post(f"{BASE_URL}/predict_batch", json=data)
results = response.json()

# Convertir a DataFrame para análisis
df = pd.DataFrame(results['predicciones'])
print(df.head())

# Top 10 mayores probabilidades
top_10 = df.nlargest(10, 'probabilidad_compra')
print("\nTop 10 predicciones:")
print(top_10[['cliente_id', 'producto_id', 'probabilidad_compra']])

# 4. Generar recomendaciones personalizadas
data = {
    "cliente_id": 254403,
    "semana": 53,
    "top_n": 5
}
response = requests.post(f"{BASE_URL}/recommend", json=data)
recommendations = response.json()

print(f"\nRecomendaciones para cliente {recommendations['cliente_id']}:")
print(f"Productos evaluados: {recommendations['n_productos_evaluados']}")
for rec in recommendations['recomendaciones']:
    print(f"{rec['ranking']}. Producto {rec['producto_id']} - {rec['probabilidad_compra']:.2%}")
    print(f"   {rec['categoria']} - {rec['marca']} ({rec['segmento']})")

# 5. Obtener predicciones de Airflow
response = requests.get(f"{BASE_URL}/latest_predictions")
airflow_preds = response.json()
print(f"\nPredicciones de Airflow: {airflow_preds['n_predicciones']}")
```

## Documentación Interactiva (Swagger)

La API incluye documentación interactiva generada automáticamente:

1. Ve a http://localhost:8000/docs
2. Verás todos los endpoints disponibles
3. Puedes probar los endpoints directamente desde el navegador:
   - Click en un endpoint
   - Click en "Try it out"
   - Completa los parámetros
   - Click en "Execute"
   - Ve la respuesta en tiempo real

**Endpoints disponibles:**
- `GET /` - Información general
- `GET /health` - Estado de la API
- `GET /model_info` - Información del modelo
- `POST /predict` - Predicción individual
- `POST /recommend` - Recomendaciones personalizadas de productos (NUEVO)
- `POST /predict_batch` - Predicción en lote
- `GET /latest_predictions` - Predicciones de Airflow

## Frontend (Gradio)

La interfaz web proporciona 4 pestañas:

1. **Predicción**: Predice probabilidad de compra para cliente-producto específico
   - Inputs: Cliente ID, Producto ID, Semana (opcional)
   - Output: Probabilidad, interpretación, info del cliente/producto

2. **Recomendaciones**: Sistema de recomendación personalizado
   - Inputs: Cliente ID, Semana (opcional), Top N (slider 1-20)
   - Output: Ranking de productos ordenados por probabilidad
   - Muestra: categoría, marca, sub-categoría, segmento para cada producto

3. **Estado del Sistema**: Monitoreo de la aplicación
   - Estado del backend (healthy/unhealthy)
   - Modelo cargado (sí/no)
   - Datos cargados (sí/no)
   - Timestamp de última verificación

4. **Modelo**: Información del modelo en producción
   - Tipo de modelo (DecisionTreeClassifier)
   - Path del archivo .pkl
   - Número de features
   - Hiperparámetros (max_depth, min_samples_split, etc.)

**Acceso**: http://localhost:7860

## Estructura del Proyecto

```
app/
├── backend/
│   ├── main.py              # Código principal de FastAPI (7 endpoints)
│   ├── transformers.py      # Transformadores custom para pipeline
│   ├── helper_functions.py  # Módulo para compatibilidad con Airflow
│   ├── Dockerfile           # Imagen Docker del backend
│   └── requirements.txt     # Dependencias Python
│
├── frontend/
│   ├── front.py             # Interfaz Gradio (4 pestañas)
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.yml       # Orquestación de servicios
└── README.md               # Esta documentación
```

## Volúmenes Compartidos

El `docker-compose.yml` monta volúmenes compartidos con Airflow:

```yaml
volumes:
  # Modelos entrenados (read-only)
  - ../airflow/data/models:/app/models:ro
  
  # Predicciones generadas (read-only)
  - ../airflow/data/predictions:/app/predictions:ro
  
  # Datos base (read-only)
  - ../airflow/data/raw:/app/data:ro
```

**Ventajas:**
- El backend siempre usa el modelo más reciente
- No necesitas copiar archivos manualmente
- Actualización automática cuando Airflow reentrena
- Solo lectura (`:ro`) para seguridad

## Troubleshooting

### Problema: "Modelo no cargado"

**Causa**: No hay modelo en `../airflow/data/models/`

**Solución**:
```bash
# 1. Verifica que Airflow haya generado el modelo
ls -lh ../airflow/data/models/modelo_*.pkl

# 2. Si no existe, ejecuta el pipeline de Airflow
cd ../airflow
docker compose exec airflow-scheduler airflow dags trigger sodai_ml_pipeline

# 3. Espera a que termine (puedes ver progreso en http://localhost:8080)

# 4. Reinicia la app
cd ../app
docker compose restart backend
```

### Problema: "Datos históricos no disponibles"

**Causa**: No hay archivos parquet en `../airflow/data/raw/`

**Solución**:
```bash
# Verifica que existan los parquets
ls -lh ../airflow/data/raw/*.parquet

# Deberías ver:
# - transacciones.parquet
# - productos.parquet
# - clientes.parquet
```

### Problema: Backend no inicia

```bash
# Ver logs detallados
docker compose logs backend

# Verificar que Airflow esté corriendo
cd ../airflow
docker compose ps

# Reiniciar servicios
cd ../app
docker compose restart
```

### Problema: Error 500 en predicciones

**Causa**: Incompatibilidad de features o preprocesamiento

**Solución**:
- Verifica que el modelo sea compatible
- Revisa los logs del backend: `docker compose logs backend`
- Asegúrate de que los datos base estén completos

## Actualización del Modelo

Cuando Airflow reentrena el modelo:

```bash
# 1. Airflow genera nuevo modelo
#    airflow/data/models/modelo_20251027_123456.pkl

# 2. La app detecta automáticamente el cambio al reiniciar
docker compose restart backend

# 3. Verifica el nuevo modelo
curl http://localhost:8000/model_info
```

**Nota**: El backend carga el modelo más reciente alfabéticamente, por eso es importante el formato `modelo_YYYYMMDD_HHMMSS.pkl`.

## Detener la Aplicación

```bash
# Detener servicios
docker compose down

# Detener y eliminar volúmenes (limpieza completa)
docker compose down -v
```

## Próximos Pasos

1. **Caché de Recomendaciones**: Redis para cachear recomendaciones frecuentes
2. **Autenticación**: JWT tokens para proteger endpoints
3. **Rate limiting**: Limitar requests por IP
4. **Batch asíncrono**: Celery para predicciones grandes
5. **Monitoreo**: Prometheus + Grafana para métricas en tiempo real
6. **Exportación de recomendaciones**: CSV/Excel de recomendaciones batch
7. **Filtros en recomendaciones**: Por categoría, marca, segmento
8. **A/B Testing**: Comparar diferentes versiones del modelo

## Contacto

**Proyecto**: SodAI Drinks - Entrega 2  
**Curso**: Laboratorio MDS 2025-2

---

*Este README documenta la API de predicciones que consume el modelo entrenado por el pipeline de Airflow.*
