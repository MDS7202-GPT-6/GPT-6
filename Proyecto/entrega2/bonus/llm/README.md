# 🤖 SodAI Drinks - Chatbot Conversacional (Bonus LLM)

Sistema de chatbot conversacional impulsado por **Groq API** (Llama 3.1) para responder preguntas sobre datos de clientes, productos y transacciones de SodAI Drinks.

## 📋 Descripción

Este chatbot permite interactuar de forma natural con los datos del negocio mediante:
- 💬 Conversación en lenguaje natural
- 🔍 Consultas específicas sobre clientes, productos y transacciones
- 📊 Análisis de ventas y tendencias
- 🎯 Respuestas basadas en datos reales del dataset

**Tecnologías:**
- **Backend:** FastAPI + Groq API (LLM)
- **Frontend:** Gradio (interfaz de chat)
- **Modelo:** Llama 3.1 8B Instant (vía Groq)
- **Datos:** Pandas (transacciones, clientes, productos)

---

## 🚀 Instalación y Configuración

### Requisitos Previos

1. **Docker y Docker Compose** instalados
2. **Cuenta de Groq** (gratuita)
3. **Datos:** Los archivos parquet en `../../airflow/data/raw/`

### Paso 1: Obtener API Key de Groq

1. Ve a [https://console.groq.com/](https://console.groq.com/)
2. Crea una cuenta gratuita (si no tienes)
3. Ve a **API Keys** en el menú
4. Crea una nueva API key
5. Copia la key (la necesitarás en el siguiente paso)

### Paso 2: Configurar Variables de Entorno

```bash
# En el directorio bonus/llm/
cp .env.example .env
```

Edita el archivo `.env` y pega tu API key:

```bash
# API Key de Groq (REQUERIDO)
GROQ_API_KEY=tu_api_key_real_aqui

# Configuración del modelo
GROQ_MODEL=llama-3.1-8b-instant
MAX_TOKENS=1024
TEMPERATURE=0.7
```

⚠️ **Importante:** 
- Nunca compartas tu API key
- Nunca hagas commit del archivo `.env` (ya está en `.gitignore`)
- Solo edita `.env.example` si quieres cambiar los valores por defecto

### Paso 3: Iniciar el Sistema

```bash
# Desde el directorio bonus/llm/
docker compose up --build
```

Esto iniciará:
- **Backend (FastAPI):** http://localhost:8002
- **Frontend (Gradio):** http://localhost:7862

### Paso 4: Usar el Chatbot

Abre tu navegador en: **http://localhost:7862**

---

## 💬 Uso del Chatbot

### Ejemplos de Preguntas

**Consultas Generales:**
```
"¿Cuántos clientes hay en el dataset?"
"¿Cuántos productos únicos existen?"
"¿Cuántas transacciones hay en total?"
```

**Análisis de Ventas:**
```
"¿Cuál es el producto más vendido?"
"¿Qué categoría tiene más ventas?"
"¿Cuántas marcas diferentes hay?"
```

**Información de Clientes:**
```
"Dame información del cliente 1000"
"¿Cuántas transacciones ha hecho el cliente 500?"
"¿Qué ha comprado el cliente 250?"
```

**Exploración de Productos:**
```
"¿Qué categorías de productos hay?"
"Muéstrame información sobre las marcas"
"¿Cuáles son los productos más populares?"
```

### Funcionalidades del Frontend

1. **💬 Chat:** Interfaz principal de conversación
   - Escribe tu pregunta
   - Presiona Enter o "Enviar"
   - Botones de ejemplo para probar rápidamente
   - Limpiar chat (solo UI)
   - Resetear conversación (borra historial en backend)

2. **📊 Estado del Sistema:** Monitoreo del backend
   - Health check del servicio
   - Estadísticas del dataset
   - Información de configuración

---

## 🏗️ Arquitectura

```
bonus/llm/
├── backend/
│   ├── main.py              # FastAPI + Groq integration
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── chat.py             # Gradio chat interface
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml       # Orquestación
├── .env                     # Variables de entorno (tu API key)
├── .env.example            # Template de configuración
├── .gitignore              # Protege .env
└── README.md               # Esta documentación
```

### Flujo de Datos

```
Usuario → Frontend (Gradio) → Backend (FastAPI) → Groq API (LLM)
                                        ↓
                                  Datos (Pandas)
                                        ↓
                            Respuesta con contexto
```

### Backend Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información del servicio |
| `/health` | GET | Estado del sistema |
| `/stats` | GET | Estadísticas del dataset |
| `/chat` | POST | Enviar mensaje al chatbot |
| `/reset` | POST | Resetear conversación |

---

## 🔧 API del Backend

### POST /chat

Envía un mensaje al chatbot y recibe una respuesta.

**Request:**
```json
{
  "message": "¿Cuántos clientes hay?",
  "reset_conversation": false
}
```

**Response:**
```json
{
  "response": "Hay 2,500 clientes únicos en el dataset...",
  "data_summary": {
    "total_transacciones": 50000,
    "total_clientes": 2500,
    "total_productos": 971
  },
  "timestamp": "2025-01-30T10:30:00"
}
```

### Ejemplo con Python

```python
import requests

# Chat simple
response = requests.post(
    "http://localhost:8002/chat",
    json={
        "message": "¿Cuál es el producto más vendido?",
        "reset_conversation": False
    }
)

print(response.json()["response"])
```

### Ejemplo con cURL

```bash
curl -X POST "http://localhost:8002/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Dame información del cliente 1000",
    "reset_conversation": false
  }'
```

---

## 🎨 Personalización

### Cambiar el Modelo LLM

En `.env`:
```bash
# Modelos disponibles en Groq:
GROQ_MODEL=llama-3.1-8b-instant      # Rápido (recomendado)
# GROQ_MODEL=llama-3.1-70b-versatile # Más potente pero lento
# GROQ_MODEL=mixtral-8x7b-32768      # Contexto largo
```

### Ajustar Temperatura

```bash
TEMPERATURE=0.7  # Default (creativo)
# TEMPERATURE=0.3  # Más determinista
# TEMPERATURE=1.0  # Muy creativo
```

### Ajustar Longitud de Respuestas

```bash
MAX_TOKENS=1024  # Default
# MAX_TOKENS=2048  # Respuestas más largas
# MAX_TOKENS=512   # Respuestas más cortas
```

---

## 🐛 Troubleshooting

### Error: "Groq API no configurada"

**Problema:** No has configurado tu API key.

**Solución:**
```bash
# 1. Copia el template
cp .env.example .env

# 2. Edita .env y pega tu API key
nano .env  # o cualquier editor

# 3. Reinicia el contenedor
docker compose restart backend
```

### Error: "Datos no cargados"

**Problema:** Los archivos parquet no están en la ruta esperada.

**Solución:**
```bash
# Verifica que existan:
ls ../../airflow/data/raw/transacciones.parquet
ls ../../airflow/data/raw/productos.parquet
ls ../../airflow/data/raw/clientes.parquet

# Si no existen, ejecuta el pipeline de Airflow primero
```

### Error de Conexión del Frontend

**Problema:** Frontend no puede conectar con backend.

**Solución:**
```bash
# Verifica que ambos contenedores estén corriendo
docker ps | grep llm

# Verifica que el backend esté saludable
curl http://localhost:8002/health

# Reinicia ambos servicios
docker compose restart
```

### Respuestas Lentas

**Problema:** El chatbot tarda mucho en responder.

**Solución:**
- Groq es generalmente rápido (<1s). Si es lento, puede ser:
  - Problema de red → Verifica tu conexión
  - Modelo grande → Cambia a `llama-3.1-8b-instant`
  - Límite de rate → Espera unos segundos

### Límite de API Rate Exceeded

**Problema:** Demasiadas requests en poco tiempo.

**Solución:**
- Groq free tier: ~30 requests/minuto
- Espera unos segundos entre consultas
- Considera upgrade a plan pago si necesitas más

---

## 📊 Datos del Sistema

El chatbot tiene acceso a:

**Transacciones:**
- `customer_id`: ID del cliente
- `product_id`: ID del producto
- `items`: Cantidad comprada
- `purchase_date`: Fecha de compra
- ... (otras columnas)

**Clientes:**
- `customer_id`: ID único
- `customer_type`: Tipo de cliente
- `region_id`, `zone_id`: Ubicación
- ... (otras columnas)

**Productos:**
- `product_id`: ID único
- `category`: Categoría del producto
- `brand`: Marca
- ... (otras columnas)

---

## 🔒 Seguridad

### Variables de Entorno

✅ **Hacer:**
- Usa `.env` para API keys
- Nunca hagas commit de `.env`
- Comparte solo `.env.example`

❌ **No hacer:**
- Hardcodear API keys en código
- Compartir `.env` en git
- Exponer API keys en logs

### Protección del .env

El archivo `.gitignore` ya incluye:
```
.env
*.env
```

### Verificar antes de Commit

```bash
# Asegúrate que .env NO esté en staging
git status

# Si aparece .env, agrégalo a .gitignore
echo ".env" >> .gitignore
git add .gitignore
```

---

## 🧪 Testing

### Probar el Backend

```bash
# Health check
curl http://localhost:8002/health

# Estadísticas
curl http://localhost:8002/stats

# Chat simple
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuántos clientes hay?"}'
```

### Probar el Frontend

1. Abre http://localhost:7862
2. Prueba los botones de ejemplo
3. Escribe tus propias preguntas
4. Verifica el tab "Estado del Sistema"

---

## 📈 Monitoreo

### Ver Logs

```bash
# Logs del backend
docker logs llm_backend -f

# Logs del frontend
docker logs llm_frontend -f

# Logs de ambos
docker compose logs -f
```

### Métricas de Uso

El backend registra:
- Número de consultas
- Tiempo de respuesta
- Errores
- Uso de tokens

Puedes verlos en los logs del contenedor.

---

## 🛑 Detener el Sistema

```bash
# Detener contenedores (mantiene volúmenes)
docker compose down

# Detener y eliminar todo
docker compose down -v

# Detener sin eliminar imágenes
docker compose stop
```

---

## 🚀 Despliegue en Producción

### Consideraciones

1. **API Keys:**
   - Usa secretos de Docker/Kubernetes
   - No uses `.env` en producción

2. **Límites de Groq:**
   - Free tier: 30 req/min
   - Considera upgrade para producción

3. **Escalabilidad:**
   - Usa reverse proxy (nginx)
   - Implementa rate limiting
   - Agrega cache para queries comunes

4. **Seguridad:**
   - HTTPS obligatorio
   - Autenticación para el frontend
   - Validación de inputs

---

## 📚 Recursos Adicionales

- [Groq Documentation](https://console.groq.com/docs)
- [Gradio Documentation](https://www.gradio.app/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Llama 3.1 Model Card](https://www.llama.com/)

---

## 🤝 Contribuir

Si encuentras bugs o quieres mejorar el chatbot:

1. Crea un issue describiendo el problema
2. Fork el repositorio
3. Crea una rama con tu feature
4. Haz commit de tus cambios
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto es parte del curso de MDS (Magíster en Data Science).

---

## ❓ FAQ

**P: ¿Cuánto cuesta usar Groq?**  
R: El tier gratuito incluye ~14k tokens/min. Para la mayoría de casos de uso educativo, es suficiente.

**P: ¿Puedo usar otro LLM?**  
R: Sí, puedes adaptar el código para usar OpenAI, Anthropic, o modelos locales con Ollama.

**P: ¿Los datos se envían a Groq?**  
R: Solo se envía contexto mínimo necesario (resumen de datos). Los datos completos permanecen locales.

**P: ¿Puedo hacer preguntas en inglés?**  
R: Sí, Llama 3.1 es multilingüe.

**P: ¿Cuánto tarda en responder?**  
R: Generalmente <1 segundo con Groq.

---

## 📞 Soporte

Si tienes problemas:

1. Revisa la sección **Troubleshooting**
2. Verifica los logs con `docker compose logs`
3. Consulta la documentación de Groq
4. Abre un issue en el repositorio

---

**¡Listo!** 🎉 Ahora tienes un chatbot conversacional funcionando con Groq API.

Para iniciar:
```bash
cp .env.example .env
# Edita .env con tu API key
docker compose up --build
# Abre http://localhost:7862
```
