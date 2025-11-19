from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import glob
from datetime import datetime

# ====================================================
# Configuración inicial
# ====================================================

MODELS_PATH = "/app/models"
DATA_PATH = "/app/data"

app = FastAPI(
    title="SodAI Drinks - Sistema de Recomendación",
    description="API para generar recomendaciones personalizadas de productos",
    version="1.0.0"
)

# ====================================================
# Carga de modelo y datos al inicio
# ====================================================

model = None
pipeline_preprocessor = None
transacciones = None
productos = None
clientes = None

@app.on_event("startup")
async def load_model_and_data():
    """Carga el modelo más reciente y los datos de entrada al iniciar la app"""
    global model, pipeline_preprocessor, transacciones, productos, clientes
    
    # Cargar modelo más reciente
    try:
        archivos_modelo = sorted(glob.glob(os.path.join(MODELS_PATH, "modelo_*.pkl")))
        if not archivos_modelo:
            raise FileNotFoundError(f"No se encontró modelo en {MODELS_PATH}")
        
        modelo_path = archivos_modelo[-1]
        model = joblib.load(modelo_path)
        print(f"Modelo cargado desde: {modelo_path}")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        raise
    
    # Cargar pipeline de preprocesamiento
    try:
        pipeline_path = os.path.join(MODELS_PATH, "pipeline_pp.pkl")
        if os.path.exists(pipeline_path):
            pipeline_preprocessor = joblib.load(pipeline_path)
            print(f"Pipeline de preprocesamiento cargado desde: {pipeline_path}")
        else:
            print(f"WARNING: Pipeline no encontrado en {pipeline_path}")
            pipeline_preprocessor = None
    except Exception as e:
        print(f"WARNING: Error cargando pipeline: {e}")
        pipeline_preprocessor = None
    
    # Cargar datos base
    try:
        transacciones = pd.read_parquet(os.path.join(DATA_PATH, "transacciones.parquet"))
        if 'purchase_date' in transacciones.columns:
            transacciones['purchase_date'] = pd.to_datetime(transacciones['purchase_date'])
        
        productos = pd.read_parquet(os.path.join(DATA_PATH, "productos.parquet"))
        clientes = pd.read_parquet(os.path.join(DATA_PATH, "clientes.parquet"))
        print(f"Datos cargados: {len(transacciones)} transacciones, {len(productos)} productos, {len(clientes)} clientes")
    except Exception as e:
        print(f"WARNING: No se pudieron cargar datos base: {e}")


# ====================================================
# Definición de esquemas de entrada
# ====================================================

class RecommendInput(BaseModel):
    cliente_id: int
    semana: int = None  # Si es None, usa semana actual + 1
    top_n: int = 5  # Número de recomendaciones a devolver


# ====================================================
# Funciones auxiliares
# ====================================================

def preparar_input_para_pipeline(df_input, productos_df, clientes_df):
    """
    Prepara el input con las columnas necesarias antes de pasar por el pipeline.
    Solo hace merge con productos y clientes, el resto lo hace el pipeline.
    
    Args:
        df_input: DataFrame con columnas [customer_id, product_id, Semana, Año]
        productos_df: DataFrame de productos
        clientes_df: DataFrame de clientes
    
    Returns:
        DataFrame con merge de clientes y productos listo para el pipeline
    """
    df = df_input.copy()
    
    # Merge con información de clientes y productos
    df = df.merge(clientes_df, on='customer_id', how='left')
    df = df.merge(productos_df, on='product_id', how='left')
    
    return df


# ====================================================
# Endpoints
# ====================================================

@app.get("/")
def root():
    return {
        "mensaje": "Sistema de Recomendación de SodAI Drinks",
        "version": "1.0.0",
        "modelo_cargado": model is not None,
        "endpoints": ["/recommend", "/health"]
    }


@app.get("/health")
def health_check():
    """Verifica el estado de la API y recursos cargados"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "modelo_cargado": model is not None,
        "datos_cargados": transacciones is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/recommend")
def recommend_products(data: RecommendInput):
    """
    Genera recomendaciones de productos para un cliente específico.
    
    Evalúa todos los productos disponibles y devuelve los top N con mayor
    probabilidad de compra.
    
    Ejemplo:
    {
        "cliente_id": 254403,
        "semana": 53,
        "top_n": 5
    }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    if transacciones is None or productos is None or clientes is None:
        raise HTTPException(status_code=503, detail="Datos históricos no disponibles")
    
    if pipeline_preprocessor is None:
        raise HTTPException(status_code=503, detail="Pipeline de preprocesamiento no cargado")
    
    try:
        # Validar que el cliente existe
        if data.cliente_id not in clientes['customer_id'].values:
            raise HTTPException(
                status_code=404, 
                detail=f"Cliente {data.cliente_id} no encontrado en la base de datos. "
                       f"Clientes disponibles: {clientes['customer_id'].min()} - {clientes['customer_id'].max()}"
            )
        
        # Preparar transacciones con formato correcto
        trans_copy = transacciones.copy()
        if 'Semana' not in trans_copy.columns:
            trans_copy['Semana'] = trans_copy['purchase_date'].dt.isocalendar().week
            trans_copy['Año'] = trans_copy['purchase_date'].dt.year
        
        max_semana = int(trans_copy['Semana'].max())
        max_año = int(trans_copy['Año'].max())
        
        semana = data.semana if data.semana else max_semana + 1
        año = max_año
        
        # Obtener todos los productos
        todos_productos = productos['product_id'].unique()
        n_productos = len(todos_productos)
        
        print(f"Generando recomendaciones para cliente {data.cliente_id}")
        print(f"Evaluando {n_productos} productos para semana {semana}...")
        
        # Crear DataFrame con todas las combinaciones cliente-producto
        df_input = pd.DataFrame({
            'customer_id': [data.cliente_id] * n_productos,
            'product_id': todos_productos,
            'Semana': [semana] * n_productos,
            'Año': [año] * n_productos
        })
        
        # Preparar input con merge de clientes y productos
        df_input = preparar_input_para_pipeline(df_input, productos, clientes)
        
        # Aplicar pipeline de preprocesamiento
        X_transformed = pipeline_preprocessor.transform(df_input)
        
        # Generar predicciones
        y_prob = model.predict_proba(X_transformed)[:, 1]
        
        # Crear DataFrame con resultados
        df_resultados = pd.DataFrame({
            'product_id': todos_productos,
            'probabilidad_compra': y_prob
        })
        
        # Merge con información de productos para incluir detalles
        df_resultados = df_resultados.merge(
            productos[['product_id', 'category', 'brand', 'sub_category', 'segment']], 
            on='product_id', 
            how='left'
        )
        
        # Ordenar por probabilidad descendente y tomar top N
        top_n = min(data.top_n, len(df_resultados))
        top_productos = df_resultados.nlargest(top_n, 'probabilidad_compra')
        
        # Obtener información del cliente
        cliente_data = clientes[clientes['customer_id'] == data.cliente_id].iloc[0]
        
        # Formatear respuesta
        recomendaciones = []
        for idx, row in top_productos.iterrows():
            recomendaciones.append({
                "ranking": len(recomendaciones) + 1,
                "producto_id": int(row['product_id']),
                "probabilidad_compra": float(row['probabilidad_compra']),
                "categoria": row['category'],
                "marca": row['brand'],
                "sub_categoria": row['sub_category'],
                "segmento": row['segment']
            })
        
        print(f"Recomendaciones generadas. Top producto: {recomendaciones[0]['producto_id']} ({recomendaciones[0]['probabilidad_compra']:.2%})")
        
        # calcular fecha de inicio (lunes) de la semana ISO solicitada para mayor claridad
        try:
            fecha_inicio_semana = datetime.fromisocalendar(int(año), int(semana), 1).date().isoformat()
        except Exception:
            # fallback: no calcular si hay cualquier error
            fecha_inicio_semana = None

        return {
            "cliente_id": data.cliente_id,
            "semana": int(semana),
            "año": int(año),
            "cliente_info": {
                "tipo": cliente_data['customer_type'],
                "region": int(cliente_data['region_id']),
                "zona": int(cliente_data['zone_id'])
            },
            "n_productos_evaluados": int(n_productos),
            "top_n": int(top_n),
            "recomendaciones": recomendaciones,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error generando recomendaciones: {str(e)}\n{traceback.format_exc()}")
