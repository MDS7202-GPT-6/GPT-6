# Chatbot Conversacional - SodAI Drinks

## Descripción

Chatbot conversacional que responde preguntas sobre datos de clientes, productos y transacciones usando Groq API (Llama 3.1) y análisis con pandas.

## Componentes

### Backend (FastAPI)
- **main.py**: API REST con endpoints de chat
  - Integración con Groq API (Llama 3.1 8B Instant)
  - Generación dinámica de código pandas según la pregunta
  - Ejecución segura de queries sobre datos
  - Historial de conversación
  - Documentación en `/docs`

### Frontend (Gradio)
- **chat.py**: Interfaz de chat interactiva
  - Chat con historial
  - Botones de ejemplo
  - Estado del sistema

## Requisitos Previos

1. **API Key de Groq**: Obtén una gratis en [console.groq.com](https://console.groq.com/)
2. **Datos**: Los archivos parquet deben estar en `../../airflow/data/raw/`

## Configuración

### 1. Obtener API Key de Groq

1. Crea cuenta gratuita en [console.groq.com](https://console.groq.com/)
2. Ve a API Keys y crea una nueva
3. Copia la key

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y pega tu API key:

```bash
GROQ_API_KEY=tu_api_key_aqui
GROQ_MODEL=llama-3.1-8b-instant
TEMPERATURE=0.7
MAX_TOKENS=1024
```

## Ejecución con Docker

### 1. Construir y levantar los servicios

Desde el directorio `llm/`:

```bash
docker compose up --build
```

Esto iniciará dos contenedores:
- **llm_backend**: API FastAPI en puerto 8002
- **llm_frontend**: Interfaz Gradio en puerto 7862

### 2. Acceder a las interfaces

- **Backend API**: http://localhost:8002
- **Documentación API**: http://localhost:8002/docs
- **Frontend Chat**: http://localhost:7862

## Uso del Chatbot

Ejemplos de preguntas:
- "¿Cuántos clientes hay en el dataset?"
- "¿Cuál es el producto más vendido?"
- "Dame información del cliente 1000"
- "¿Qué categorías de productos hay?"
- "¿Cuántas transacciones ha hecho el cliente 500?"

## Uso de la API

La API expone los siguientes endpoints:

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información general de la API |
| `/health` | GET | Estado del sistema |
| `/stats` | GET | Estadísticas del dataset |
| `/chat` | POST | Enviar mensaje al chatbot |
| `/reset` | POST | Resetear historial de conversación |

### Ejemplo de uso

```bash
curl -X POST "http://localhost:8002/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Cuántos clientes hay?",
    "reset_conversation": false
  }'
```

## Documentación de la API

La documentación interactiva está disponible en:
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

## Estructura del Proyecto

```
llm/
├── backend/
│   ├── main.py              # FastAPI + Groq integration
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── chat.py              # Interfaz Gradio
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.yml       # Orquestación de servicios
├── .env                     # Variables de entorno (API key)
├── .env.example             # Template de configuración
└── README.md
```

## Cómo Funciona

1. El usuario hace una pregunta en lenguaje natural
2. El LLM (Llama 3.1) analiza la pregunta y genera código pandas
3. El backend ejecuta el código de forma segura sobre los datos
4. El LLM usa el resultado para generar una respuesta natural
5. La respuesta se muestra en el frontend

## Detener la Aplicación

```bash
docker compose down
```

---

**Proyecto**: SodAI Drinks - Laboratorio MDS 2025-2

**Nota**: Nunca hagas commit del archivo `.env` que contiene tu API key. El archivo `.gitignore` ya está configurado para protegerlo.
