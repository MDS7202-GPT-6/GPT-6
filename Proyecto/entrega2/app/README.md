# API de Predicciones - SodAI Drinks

## Descripción

Aplicación web para el sistema de predicción de compra semanal de productos. La aplicación carga el modelo de machine learning entrenado por el pipeline de Airflow y expone una API REST junto con una interfaz gráfica para realizar predicciones.

## Componentes

### Backend (FastAPI)
- **main.py**: API REST con 5 endpoints para realizar predicciones
  - Carga automática del modelo más reciente desde Airflow (`modelo_*.pkl`)
  - Carga del pipeline de preprocesamiento completo (`pipeline_pp.pkl`)
  - Carga de datos históricos en formato parquet (transacciones, productos, clientes)
  - El pipeline cargado se encarga de todo el preprocesamiento (features, clustering, encoding)
  - Documentación interactiva en `/docs`

### Frontend (Gradio)
- **front.py**: Interfaz web con 3 pestañas
  - **Predicción**: Predice probabilidad de compra para un cliente-producto específico
  - **Estado del Sistema**: Monitorea el estado del backend, modelo y datos
  - **Modelo**: Muestra información del modelo cargado (tipo, hiperparámetros, features)

## Requisitos Previos

El pipeline de Airflow debe haberse ejecutado al menos una vez para generar:
- Modelo entrenado: `../airflow/data/models/modelo_*.pkl`
- Pipeline de preprocesamiento: `../airflow/data/models/pipeline_pp.pkl`
- Datos en formato parquet: `../airflow/data/raw/*.parquet`

Si no has ejecutado Airflow, hazlo primero:
```bash
cd ../airflow
docker compose up -d
```

## Ejecución con Docker

### 1. Construir y levantar los servicios

Desde el directorio `app/`:

```bash
docker compose build
docker compose up -d
```

Esto iniciará dos contenedores:
- **sodai_backend**: API FastAPI en puerto 8000
- **sodai_frontend**: Interfaz Gradio en puerto 7860

### 2. Verificar que los servicios estén corriendo

```bash
docker compose ps
```

Deberías ver ambos contenedores con estado "Up".

### 3. Verificar logs

```bash
docker compose logs backend
```

Busca estas líneas que confirman que todo se cargó correctamente:
- "Modelo cargado desde: /app/models/modelo_*.pkl"
- "Pipeline de preprocesamiento cargado desde: /app/models/pipeline_pp.pkl"
- "Datos cargados: X transacciones, Y productos, Z clientes"

### 4. Acceder a las interfaces

- **Backend API**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs
- **Frontend Web**: http://localhost:7860

## Uso de la API

La API expone los siguientes endpoints:

### Endpoints Disponibles

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información general de la API |
| `/health` | GET | Estado del sistema (modelo, datos, pipeline) |
| `/model_info` | GET | Detalles del modelo cargado |
| `/predict` | POST | Predicción individual para cliente-producto |
| `/latest_predictions` | GET | Predicciones históricas generadas por Airflow |

### Ejemplos de Uso

#### 1. Verificar estado del sistema
```bash
curl http://localhost:8000/health
```

#### 2. Predicción individual
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cliente_id": 254403,
    "producto_id": 34092,
    "semana": 53
  }'
```

Respuesta:
```json
{
  "cliente_id": 254403,
  "producto_id": 34092,
  "semana": 53,
  "año": 2024,
  "prediccion": 1,
  "probabilidad_compra": 0.8234,
  "interpretacion": "Comprará",
  "cliente_info": {
    "tipo": "RETAIL",
    "region": 1,
    "zona": 5
  },
  "producto_info": {
    "categoria": "BEBIDAS CARBONATADAS",
    "marca": "Brand 24"
  }
}
```

## Documentación de la API

La documentación interactiva está disponible en:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Desde Swagger UI puedes probar todos los endpoints directamente desde el navegador.

## Estructura del Proyecto

```
app/
├── backend/
│   ├── main.py              # API FastAPI con 5 endpoints
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── front.py             # Interfaz Gradio (3 pestañas)
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.yml       # Orquestación de servicios
└── README.md
```

## Detener la Aplicación

```bash
docker compose down
```

---

**Proyecto**: SodAI Drinks - Laboratorio MDS 2025-2
