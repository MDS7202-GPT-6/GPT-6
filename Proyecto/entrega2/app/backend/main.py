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
PREDICTIONS_PATH = "/app/predictions"
DATA_PATH = "/app/data"

app = FastAPI(
    title="SodAI Drinks - API de Predicciones",
    description="API para exponer el modelo de predicción de compra semanal.",
    version="2.0.0"
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
            print(f"WARNING: Pipeline no encontrado en {pipeline_path}, usando preprocesamiento simplificado")
            pipeline_preprocessor = None
    except Exception as e:
        print(f"WARNING: Error cargando pipeline: {e}, continuando sin pipeline")
        pipeline_preprocessor = None
    
    # Cargar datos base
    try:
        transacciones = pd.read_parquet(os.path.join(DATA_PATH, "transacciones.parquet"))
        # Convertir purchase_date a datetime si es necesario
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

class PredictSingleInput(BaseModel):
    cliente_id: int
    producto_id: int
    semana: int = None  # Si es None, usa semana actual + 1


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
        "mensaje": "API SodAI Drinks funcionando correctamente.",
        "version": "2.0.0",
        "modelo_cargado": model is not None,
        "endpoints": ["/predict", "/health", "/model_info", "/latest_predictions"]
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


@app.get("/model_info")
def model_info():
    """Información sobre el modelo cargado"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    modelo_files = sorted(glob.glob(os.path.join(MODELS_PATH, "modelo_*.pkl")))
    modelo_actual = modelo_files[-1] if modelo_files else "Desconocido"
    
    return {
        "modelo_path": modelo_actual,
        "tipo_modelo": str(type(model).__name__),
        "parametros": model.get_params() if hasattr(model, 'get_params') else {},
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Desconocido"
    }


@app.post("/predict")
def predict_single(data: PredictSingleInput):
    """
    Predice la probabilidad de compra para un par cliente-producto.
    
    Requiere transacciones históricas para calcular features de frecuencia.
    
    Ejemplo:
    {
        "cliente_id": 254403,
        "producto_id": 34092,
        "semana": 53
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
        
        # Validar que el producto existe
        if data.producto_id not in productos['product_id'].values:
            raise HTTPException(
                status_code=404, 
                detail=f"Producto {data.producto_id} no encontrado en la base de datos. "
                       f"Productos disponibles: {productos['product_id'].min()} - {productos['product_id'].max()}"
            )
        
        # Preparar transacciones con formato correcto
        trans_copy = transacciones.copy()
        if 'Semana' not in trans_copy.columns:
            trans_copy['Semana'] = trans_copy['purchase_date'].dt.isocalendar().week
            trans_copy['Año'] = trans_copy['purchase_date'].dt.year
        
        max_semana = int(trans_copy['Semana'].max())
        max_año = int(trans_copy['Año'].max())
        
        semana = data.semana if data.semana else max_semana + 1
        año = max_año  # Asumimos mismo año
        
        # Crear DataFrame con formato exacto que espera el pipeline
        df_input = pd.DataFrame([{
            'customer_id': data.cliente_id,
            'product_id': data.producto_id,
            'Semana': semana,
            'Año': año
        }])
        
        # Merge con información de clientes y productos (IGUAL QUE EN AIRFLOW)
        df_input = df_input.merge(clientes, on='customer_id', how='left')
        df_input = df_input.merge(productos, on='product_id', how='left')
        
        # El pipeline cargado tiene un FunctionTransformer que captura df_transacciones
        # en su closure, por lo que transform() debería funcionar directamente
        X_transformed = pipeline_preprocessor.transform(df_input)
        
        # Predecir
        y_pred = model.predict(X_transformed)
        y_prob = model.predict_proba(X_transformed)[:, 1]
        
        # Obtener info adicional
        cliente_data = clientes[clientes['customer_id'] == data.cliente_id].iloc[0]
        producto_data = productos[productos['product_id'] == data.producto_id].iloc[0]
        
        return {
            "cliente_id": data.cliente_id,
            "producto_id": data.producto_id,
            "semana": int(semana),
            "año": int(año),
            "prediccion": int(y_pred[0]),
            "probabilidad_compra": float(y_prob[0]),
            "interpretacion": "Comprará" if y_pred[0] == 1 else "No comprará",
            "cliente_info": {
                "tipo": cliente_data['customer_type'],
                "region": int(cliente_data['region_id']),
                "zona": int(cliente_data['zone_id'])
            },
            "producto_info": {
                "categoria": producto_data['category'],
                "marca": producto_data['brand']
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}\n{traceback.format_exc()}")


@app.get("/latest_predictions")
def get_latest_predictions():
    """Devuelve las últimas predicciones generadas por el pipeline de Airflow"""
    try:
        pred_files = sorted(glob.glob(os.path.join(PREDICTIONS_PATH, "predicciones_*.csv")))
        if not pred_files:
            raise HTTPException(status_code=404, detail="No hay predicciones disponibles")
        
        latest_pred = pd.read_csv(pred_files[-1])
        
        return {
            "archivo": os.path.basename(pred_files[-1]),
            "n_predicciones": len(latest_pred),
            "top_10_probabilidades": latest_pred.nlargest(10, 'probabilidad_compra').to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo predicciones: {str(e)}")