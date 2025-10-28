# Sistema de Recomendación - SodAI Drinks

## Descripción

Sistema de recomendación personalizado que genera las mejores 5 recomendaciones de productos para cualquier cliente basándose en probabilidades predictivas del modelo de machine learning.

## Características

- **Recomendaciones personalizadas**: Evalúa todos los productos disponibles (~971) para un cliente específico
- **Top N configurable**: Permite seleccionar entre 1 y 20 productos recomendados
- **Scoring detallado**: Cada recomendación incluye:
  - Probabilidad de compra
  - Categoría del producto
  - Marca
  - Sub-categoría
  - Segmento
- **Backend/Frontend separados**: Arquitectura de microservicios dockerizada
- **Interfaz amigable**: Frontend con Gradio para usuarios no técnicos

## Arquitectura

```
┌─────────────────────────────────────────────────┐
│              Airflow Pipeline                   │
│  (Genera modelo y datos)                        │
│                                                 │
│  Outputs:                                       │
│  • /data/models/modelo_*.pkl                   │
│  • /data/raw/*.parquet                         │
└─────────────┬───────────────────────────────────┘
              │ (volúmenes compartidos)
              ▼
┌─────────────────────────────────────────────────┐
│           Sistema de Recomendación              │
│                                                 │
│  ┌──────────────┐         ┌──────────────┐    │
│  │   Backend    │◄────────│   Frontend   │    │
│  │  (FastAPI)   │         │  (Gradio)    │    │
│  │  Port 8001   │         │  Port 7861   │    │
│  └──────────────┘         └──────────────┘    │
└─────────────────────────────────────────────────┘
```

## Requisitos Previos

1. **Pipeline de Airflow ejecutado**: El modelo debe estar entrenado y guardado
2. **Docker y Docker Compose**: Para levantar los servicios

Verifica que existan:
```bash
ls -lh ../../airflow/data/models/modelo_*.pkl
ls -lh ../../airflow/data/raw/*.parquet
```

## Instalación y Ejecución

### Paso 1: Navegar al directorio

```bash
cd /path/to/Proyecto/entrega2/bonus/recsys
```

### Paso 2: Construir y levantar servicios

```bash
docker compose build
docker compose up -d
```

### Paso 3: Verificar contenedores

```bash
docker compose ps

# Salida esperada:
# NAME                STATUS              PORTS
# recsys_backend      Up (healthy)        0.0.0.0:8001->8001/tcp
# recsys_frontend     Up                  0.0.0.0:7861->7861/tcp
```

### Paso 4: Acceder a las interfaces

**Backend API (FastAPI):**
```
URL: http://localhost:8001
Documentación: http://localhost:8001/docs
```

**Frontend (Gradio):**
```
URL: http://localhost:7861
```

## Uso de la API

### Health Check

```bash
curl http://localhost:8001/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "modelo_cargado": true,
  "datos_cargados": true,
  "timestamp": "2024-10-27T00:30:00"
}
```

### Generar Recomendaciones

```bash
curl -X POST "http://localhost:8001/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "cliente_id": 254403,
    "semana": 53,
    "top_n": 5
  }'
```

**Respuesta:**
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
    ...
  ],
  "timestamp": "2024-10-27T00:35:00"
}
```

## Uso desde Python

```python
import requests

# Generar recomendaciones
data = {
    "cliente_id": 254403,
    "semana": 53,
    "top_n": 5
}
response = requests.post("http://localhost:8001/recommend", json=data)
result = response.json()

print(f"Cliente: {result['cliente_id']}")
print(f"Productos evaluados: {result['n_productos_evaluados']}")
print("\nTop 5 Recomendaciones:")
for rec in result['recomendaciones']:
    print(f"{rec['ranking']}. Producto {rec['producto_id']} - {rec['probabilidad_compra']:.2%}")
    print(f"   {rec['categoria']} - {rec['marca']} ({rec['segmento']})")
```

## Interfaz Gradio

La interfaz web incluye:

1. **Tab Recomendaciones**:
   - Input: Cliente ID, Semana (opcional), Top N (slider 1-20)
   - Output: Lista de productos recomendados con detalles
   - Clasificación visual de probabilidades (MUY ALTA, ALTA, MEDIA, BAJA)

2. **Tab Estado del Sistema**:
   - Monitoreo del estado del backend
   - Verificación de modelo y datos cargados

3. **Accordion con explicación** del funcionamiento del sistema

## Metodología

El sistema funciona en 4 pasos:

1. **Análisis de perfil**: Identifica características del cliente (tipo, región, zona)
2. **Evaluación masiva**: Crea combinaciones cliente × todos_productos
3. **Scoring predictivo**: Aplica el modelo ML para obtener probabilidad de compra
4. **Ranking**: Ordena por probabilidad y devuelve top N

## Estructura del Proyecto

```
recsys/
├── backend/
│   ├── main.py              # API FastAPI con endpoint /recommend
│   ├── transformers.py      # Transformadores custom del pipeline
│   ├── helper_functions.py  # Módulo de compatibilidad
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── front.py             # Interfaz Gradio
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml       # Orquestación de servicios
└── README.md               # Esta documentación
```

## Volúmenes Compartidos

El sistema accede a:

```yaml
volumes:
  # Modelos entrenados por Airflow (read-only)
  - ../../airflow/data/models:/app/models:ro
  
  # Datos base (read-only)
  - ../../airflow/data/raw:/app/data:ro
```

## Puertos

- **Backend**: 8001
- **Frontend**: 7861

Estos puertos son diferentes de la aplicación principal (app/) para evitar conflictos.

## Casos de Uso

1. **Campañas de Marketing**: Personalizar ofertas por cliente
2. **App Móvil**: Mostrar productos sugeridos al abrir sesión
3. **Email Marketing**: Enviar recomendaciones semanales
4. **Optimización de Inventario**: Priorizar stock según demanda predictiva

## Troubleshooting

### Problema: "Modelo no cargado"

```bash
# Verificar que exista el modelo
ls -lh ../../airflow/data/models/modelo_*.pkl

# Si no existe, ejecutar pipeline de Airflow
cd ../../airflow
docker compose exec airflow-scheduler airflow dags trigger sodai_ml_pipeline

# Reiniciar recsys
cd ../bonus/recsys
docker compose restart backend
```

### Problema: Puerto 8001 ya en uso

Editar `docker-compose.yml` y cambiar el puerto:
```yaml
ports:
  - "8002:8001"  # Usar 8002 en lugar de 8001
```

### Problema: Error de conexión frontend → backend

```bash
# Ver logs
docker compose logs backend
docker compose logs frontend

# Verificar red
docker network inspect recsys_recsys_network
```

## Detener el Sistema

```bash
# Detener servicios
docker compose down

# Detener y limpiar
docker compose down -v
```

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
