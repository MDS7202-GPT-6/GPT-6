from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
from datetime import datetime
from groq import Groq
import json
import re

# ====================================================
# Configuración inicial
# ====================================================

DATA_PATH = "/app/data"

app = FastAPI(
    title="SodAI Drinks - Chatbot Conversacional",
    description="Chatbot para responder preguntas sobre datos de transacciones, clientes y productos",
    version="1.0.0"
)

# Variables globales
transacciones = None
productos = None
clientes = None
groq_client = None
conversation_history = []

# ====================================================
# Carga de datos y modelo al inicio
# ====================================================

@app.on_event("startup")
async def load_data_and_model():
    """Carga los datos y configura el cliente de Groq"""
    global transacciones, productos, clientes, groq_client
    
    # Cargar API key de Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        print("WARNING: GROQ_API_KEY no configurada. El chatbot no funcionará.")
        print("Por favor, configura tu API key en el archivo .env")
    else:
        groq_client = Groq(api_key=groq_api_key)
        print("Cliente Groq inicializado correctamente")
    
    # Cargar datos
    try:
        transacciones = pd.read_parquet(os.path.join(DATA_PATH, "transacciones.parquet"))
        if 'purchase_date' in transacciones.columns:
            transacciones['purchase_date'] = pd.to_datetime(transacciones['purchase_date'])
        
        productos = pd.read_parquet(os.path.join(DATA_PATH, "productos.parquet"))
        clientes = pd.read_parquet(os.path.join(DATA_PATH, "clientes.parquet"))
        
        print(f"Datos cargados: {len(transacciones)} transacciones, {len(productos)} productos, {len(clientes)} clientes")
    except Exception as e:
        print(f"ERROR cargando datos: {e}")
        raise


# ====================================================
# Funciones auxiliares
# ====================================================

def get_data_summary():
    """Retorna un resumen de los datos para el contexto del LLM"""
    summary = {
        "total_transacciones": len(transacciones),
        "total_clientes": len(clientes),
        "total_productos": len(productos),
        "categorias": productos['category'].unique().tolist(),
        "marcas_ejemplo": productos['brand'].unique().tolist()[:10],
        "tipos_cliente": clientes['customer_type'].unique().tolist(),
    }
    return summary


def execute_pandas_query(code: str) -> str:
    """
    Ejecuta código pandas generado por el LLM de forma segura.
    El código tiene acceso a: transacciones, productos, clientes, pd
    """
    try:
        print(f"\n{'='*60}")
        print(f"[🔍 EJECUTANDO QUERY GENERADA POR EL LLM]")
        print(f"{'='*60}")
        print(f"{code}")
        print(f"{'='*60}\n")
        
        # Crear namespace seguro con acceso a datos y pandas
        safe_globals = {
            'pd': pd,
            'transacciones': transacciones,
            'productos': productos,
            'clientes': clientes,
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'round': round,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
        }
        
        safe_locals = {}
        
        # Ejecutar código
        exec(code, safe_globals, safe_locals)
        
        # Obtener resultado
        result = safe_locals.get('result', None)
        
        if result is None:
            return "⚠️ La query no retornó ningún resultado (debe asignar a 'result')"
        
        # Convertir resultado a string legible
        if isinstance(result, pd.DataFrame):
            result_str = result.to_string()
        elif isinstance(result, pd.Series):
            result_str = result.to_string()
        else:
            result_str = str(result)
        
        print(f"[✅ RESULTADO DE LA QUERY]:")
        print(f"{result_str}\n")
        
        return result_str
        
    except Exception as e:
        error_msg = f"❌ Error ejecutando query: {str(e)}"
        print(f"[ERROR]: {error_msg}\n")
        return error_msg


# ====================================================
# Definición de esquemas
# ====================================================

class ChatMessage(BaseModel):
    message: str
    reset_conversation: bool = False

class ChatResponse(BaseModel):
    response: str
    data_summary: Optional[dict] = None
    timestamp: str


# ====================================================
# Endpoints
# ====================================================

@app.get("/")
def root():
    return {
        "mensaje": "Chatbot Conversacional de SodAI Drinks",
        "version": "1.0.0",
        "groq_configurado": groq_client is not None,
        "datos_cargados": transacciones is not None,
        "endpoints": ["/chat", "/health", "/reset", "/stats"]
    }


@app.get("/health")
def health_check():
    """Verifica el estado del chatbot"""
    return {
        "status": "healthy" if groq_client is not None else "unhealthy",
        "groq_configurado": groq_client is not None,
        "datos_cargados": transacciones is not None,
        "modelo": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
def get_stats():
    """Retorna estadísticas básicas de los datos"""
    if transacciones is None:
        raise HTTPException(status_code=503, detail="Datos no cargados")
    
    return {
        "total_transacciones": len(transacciones),
        "total_clientes": len(clientes),
        "total_productos": len(productos),
        "categorias": productos['category'].nunique(),
        "marcas": productos['brand'].nunique(),
        "fecha_min": transacciones['purchase_date'].min().isoformat() if 'purchase_date' in transacciones.columns else None,
        "fecha_max": transacciones['purchase_date'].max().isoformat() if 'purchase_date' in transacciones.columns else None,
    }


@app.post("/reset")
def reset_conversation():
    """Resetea el historial de conversación"""
    global conversation_history
    conversation_history = []
    return {"mensaje": "Historial de conversación reseteado"}


@app.post("/chat", response_model=ChatResponse)
async def chat(data: ChatMessage):
    """
    Endpoint principal del chatbot.
    El LLM genera código pandas dinámicamente según la pregunta.
    """
    global conversation_history
    
    if groq_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Groq API no configurada. Por favor configura GROQ_API_KEY en el archivo .env"
        )
    
    if transacciones is None:
        raise HTTPException(status_code=503, detail="Datos no cargados")
    
    # Resetear conversación si se solicita
    if data.reset_conversation:
        conversation_history = []
    
    try:
        print(f"\n{'#'*60}")
        print(f"[💬 NUEVA PREGUNTA]: {data.message}")
        print(f"{'#'*60}\n")
        
        # Preparar contexto para el LLM
        data_summary = get_data_summary()
        
        # System prompt con instrucciones para generar código pandas
        system_prompt = f"""Eres un asistente experto en análisis de datos para SodAI Drinks, una empresa de bebidas.

DATOS DISPONIBLES:
Tienes acceso a 3 DataFrames de pandas:

1. **transacciones** ({data_summary['total_transacciones']:,} registros):
   - customer_id: ID del cliente
   - product_id: ID del producto  
   - order_id: ID de la orden
   - purchase_date: Fecha de compra (datetime)
   - items: Cantidad de items comprados

2. **productos** ({data_summary['total_productos']:,} registros):
   - product_id: ID del producto
   - brand: Marca del producto
   - category: Categoría ({', '.join(data_summary['categorias'])})
   - sub_category: Sub-categoría
   - segment: Segmento (LOW, MEDIUM, HIGH, PREMIUM)
   - package: Tipo de empaque
   - size: Tamaño

3. **clientes** ({data_summary['total_clientes']:,} registros):
   - customer_id: ID del cliente
   - customer_type: Tipo de cliente ({', '.join(data_summary['tipos_cliente'])})
   - region_id: ID de región
   - zone_id: ID de zona
   - num_deliver_per_week: Entregas por semana
   - num_visit_per_week: Visitas por semana
   - X, Y: Coordenadas geográficas

REGLAS:

1. **SIEMPRE** que necesites consultar o analizar datos, genera código pandas dentro de bloques de código.

2. **Formato obligatorio** para código:
```python
# Tu análisis en pandas
result = <valor_final>
```

3. La variable `result` DEBE contener la respuesta final (puede ser número, string, DataFrame, lista, etc.)

4. **Ejemplos de queries correctas**:

Pregunta: "¿Cuántos clientes hay?"
```python
result = len(clientes['customer_id'].unique())
```

Pregunta: "¿Cuál es el producto más vendido?"
```python
ventas = transacciones.groupby('product_id')['items'].sum().sort_values(ascending=False)
top_id = ventas.index[0]
prod_info = productos[productos['product_id'] == top_id].iloc[0]
result = f"Producto {{top_id}}: {{prod_info['brand']}} - {{prod_info['category']}} ({{ventas.iloc[0]:.0f}} unidades)"
```

Pregunta: "¿Qué compró el cliente 100?"
```python
trans = transacciones[transacciones['customer_id'] == 100]
trans_prod = trans.merge(productos, on='product_id')
result = trans_prod[['product_id', 'brand', 'category', 'items']].to_string()
```

5. **Manejo de fechas**: Usa `pd.to_datetime()` si necesitas filtrar por fechas.

6. **NO inventes datos**. Si no tienes información, genera código para obtenerla.

7. Para preguntas simples (saludos, agradecimientos), responde directamente sin código.

8. Sé conciso y preciso. Los usuarios esperan respuestas basadas en datos reales.
"""
        
        # Construir mensajes
        messages = [{"role": "system", "content": system_prompt}]
        
        # Agregar historial (últimos 3 mensajes)
        for msg in conversation_history[-3:]:
            messages.append(msg)
        
        # Agregar pregunta actual
        messages.append({"role": "user", "content": data.message})
        
        # Primera llamada a Groq (para decidir si necesita ejecutar código)
        print(f"[🤖 LLAMANDO AL LLM - Primera llamada...]")
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
        )
        
        assistant_message = chat_completion.choices[0].message.content
        
        print(f"\n{'='*60}")
        print(f"[📝 RESPUESTA DEL LLM - Primera llamada]:")
        print(f"{'='*60}")
        print(assistant_message)
        print(f"{'='*60}\n")
        
        # Buscar bloques de código python en la respuesta
        code_blocks = re.findall(r'```python\n(.*?)\n```', assistant_message, re.DOTALL)
        
        query_results = []
        
        # Ejecutar todos los bloques de código encontrados
        for code in code_blocks:
            result = execute_pandas_query(code)
            query_results.append(result)
        
        # Si se ejecutaron queries, hacer segunda llamada con resultados
        if query_results:
            results_text = "\n\n".join([f"RESULTADO DE QUERY:\n{r}" for r in query_results])
            
            print(f"[🔄 PREPARANDO SEGUNDA LLAMADA AL LLM CON LOS RESULTADOS...]")
            
            messages.append({"role": "assistant", "content": assistant_message})
            messages.append({
                "role": "user", 
                "content": f"Los resultados de las queries son:\n\n{results_text}\n\nAhora responde la pregunta original de forma natural y clara, usando estos resultados."
            })
            
            # Segunda llamada para respuesta final
            print(f"[🤖 LLAMANDO AL LLM - Segunda llamada...]")
            final_completion = groq_client.chat.completions.create(
                messages=messages,
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
            )
            
            response_text = final_completion.choices[0].message.content
            
            print(f"\n{'='*60}")
            print(f"[📝 RESPUESTA DEL LLM - Segunda llamada (FINAL)]:")
            print(f"{'='*60}")
            print(response_text)
            print(f"{'='*60}\n")
        else:
            # No se ejecutaron queries, usar respuesta directa
            print(f"[ℹ️  NO SE GENERARON QUERIES - Usando respuesta directa del LLM]")
            response_text = assistant_message
        
        # Actualizar historial
        conversation_history.append({"role": "user", "content": data.message})
        conversation_history.append({"role": "assistant", "content": response_text})
        
        print(f"\n[📤 RESPUESTA FINAL]: {response_text}\n")
        
        return ChatResponse(
            response=response_text,
            data_summary=data_summary,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Error en el chatbot: {str(e)}\n{traceback.format_exc()}"
        print(f"\n[❌ ERROR]: {error_detail}\n")
        raise HTTPException(status_code=500, detail=error_detail)
