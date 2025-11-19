# Sistema de Recomendación - SodAI Drinks

## Descripción

Sistema de recomendación personalizado que evalúa todos los productos disponibles para un cliente específico y genera un ranking basado en la probabilidad de compra predicha por el modelo de machine learning.

## Componentes

### Backend (FastAPI)
- **main.py**: API REST con 2 endpoints
  - Carga automática del modelo más reciente desde Airflow (`modelo_*.pkl`)
  - Carga del pipeline de preprocesamiento completo (`pipeline_pp.pkl`)
  - Carga de datos históricos en formato parquet (transacciones, productos, clientes)
  - El pipeline se encarga de todo el preprocesamiento (features, clustering, encoding)
  - Documentación interactiva en `/docs`

### Frontend (Gradio)
- **front.py**: Interfaz web con 2 pestañas
  - **Recomendaciones**: Genera top N productos recomendados para un cliente
  - **Estado del Sistema**: Monitorea el estado del backend, modelo y datos

## Requisitos Previos

El pipeline de Airflow debe haberse ejecutado al menos una vez para generar:
- Modelo entrenado: `../../airflow/data/models/modelo_*.pkl`
- Pipeline de preprocesamiento: `../../airflow/data/models/pipeline_pp.pkl`
- Datos en formato parquet: `../../airflow/data/raw/*.parquet`

Si no has ejecutado Airflow, hazlo primero:
```bash
cd ../../airflow
docker compose up -d
```

## Ejecución con Docker

### 1. Construir y levantar los servicios

Desde el directorio `recsys/`:

```bash
docker compose build
docker compose up -d
```

Esto iniciará dos contenedores:
- **recsys_backend**: API FastAPI en puerto 8001
- **recsys_frontend**: Interfaz Gradio en puerto 7861

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

- **Backend API**: http://localhost:8001
- **Documentación API**: http://localhost:8001/docs
- **Frontend Web**: http://localhost:7861

## Uso de la API

La API expone los siguientes endpoints:

### Endpoints Disponibles

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información general de la API |
| `/health` | GET | Estado del sistema (modelo, datos, pipeline) |
| `/recommend` | POST | Genera recomendaciones personalizadas para un cliente |

### Ejemplos de Uso

#### 1. Verificar estado del sistema
```bash
curl http://localhost:8001/health
```

#### 2. Generar recomendaciones
```bash
curl -X POST "http://localhost:8001/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "cliente_id": 254403,
    "semana": 53,
    "top_n": 5
  }'
```

Respuesta:
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
    }
  ],
  "timestamp": "2024-10-27T00:35:00"
}
```

El sistema evalúa todos los productos disponibles (~971) y devuelve los top N ordenados por probabilidad de compra.

## Documentación de la API

La documentación interactiva está disponible en:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

Desde Swagger UI puedes probar todos los endpoints directamente desde el navegador.

## Estructura del Proyecto

```
recsys/
├── backend/
│   ├── main.py              # API FastAPI con endpoint /recommend
│   ├── helper_functions.py  # Funciones ML y transformers
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── front.py             # Interfaz Gradio (2 pestañas)
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.yml       # Orquestación de servicios
└── README.md
```

## Diferencias con app/

Este sistema está diseñado específicamente para recomendaciones:

| Aspecto | app/ | recsys/ |
|---------|------|---------|
| **Propósito** | Predicción individual | Recomendaciones personalizadas |
| **Endpoints** | /predict, /health, /model_info | /recommend, /health |
| **Puerto Backend** | 8000 | 8001 |
| **Puerto Frontend** | 7860 | 7861 |
| **Funcionalidad** | Predice para cliente-producto específico | Evalúa todos los productos (~971) para un cliente |
| **Output** | Sí/No comprará + probabilidad | Top N productos ordenados por probabilidad |

## Detener la Aplicación

```bash
docker compose down
```

---

**Proyecto**: SodAI Drinks - Laboratorio MDS 2025-2

## Diferencias con app/

| Aspecto | app/ | recsys/ |
|---------|------|---------|
| **Propósito** | Predicciones individuales/batch | Recomendaciones personalizadas |
| **Endpoints** | /predict, /predict_batch, /health, /model_info | /recommend, /health |
| **Puerto Backend** | 8000 | 8001 |
| **Puerto Frontend** | 7860 | 7861 |
| **Funcionalidad** | Predecir cliente-producto específico | Evaluar todos los productos para un cliente |
| **Output** | Sí/No comprará + probabilidad | Top N productos ordenados por probabilidad |

## Tecnologías Utilizadas

- **Backend**: FastAPI 0.104.1
- **Frontend**: Gradio 4.8.0
- **Modelo**: DecisionTreeClassifier (sklearn)
- **Tracking**: MLflow
- **Orquestación**: Apache Airflow
- **Contenedorización**: Docker + Docker Compose

---

**Proyecto:** SodAI Drinks - Sistema de Recomendación  
**Curso:** MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos  
**Bonus:** 0.5 puntos
