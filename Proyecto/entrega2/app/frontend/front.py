import gradio as gr
import requests

API_URL = "http://backend:8000"

# ======================================================
# 1. Función de predicción individual
# ======================================================
def predecir(cliente_id, producto_id, semana):
    """Realiza una predicción individual"""
    data = {
        "cliente_id": int(cliente_id),
        "producto_id": int(producto_id),
        "semana": int(semana) if semana else None
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        if response.status_code == 200:
            r = response.json()
            comprara = r['prediccion'] == 1
            probabilidad = r['probabilidad_compra']
            interpretacion = r.get('interpretacion', 'Comprará' if comprara else 'No comprará')

            # Color según resultado
            texto_resultado = "SÍ COMPRARÁ" if comprara else "NO COMPRARÁ"

            # Clasificar probabilidad
            if probabilidad >= 0.7:
                nivel_prob = "MUY ALTA"
            elif probabilidad >= 0.5:
                nivel_prob = "ALTA"
            elif probabilidad >= 0.3:
                nivel_prob = "MEDIA"
            else:
                nivel_prob = "BAJA"

            result = f"""# **RESULTADO: {texto_resultado}**

---

## Detalles de la Predicción

**Cliente ID:** {r['cliente_id']} | **Producto ID:** {r['producto_id']} | **Semana:** {r['semana']} ({r.get('año', 'N/A')})

### Probabilidad de Compra: **{probabilidad:.1%}** ({nivel_prob})

---

### Información del Cliente
- **Tipo:** {r['cliente_info']['tipo']}
- **Región:** {r['cliente_info']['region']}
- **Zona:** {r['cliente_info']['zona']}

### Información del Producto
- **Categoría:** {r['producto_info']['categoria']}
- **Marca:** {r['producto_info']['marca']}

---

### Interpretación
{interpretacion} con una probabilidad del **{probabilidad:.1%}**

*Timestamp: {r['timestamp']}*
"""
            return result

        elif response.status_code == 404:
            error_detail = response.json().get('detail', 'Error desconocido')
            return f"""**Error: ID no encontrado**

{error_detail}

**Sugerencia:**
Verifica que el ID sea correcto. Puedes consultar los rangos válidos en la sección de Estado del Sistema.
"""
        else:
            error_detail = response.json().get('detail', 'Error desconocido')
            return f"""**Error en la predicción**

{error_detail}

**Nota:** El pipeline de preprocesamiento debe estar disponible.
Asegúrate de que el DAG de Airflow se haya ejecutado al menos una vez.
"""
    except Exception as e:
        return f"""**Error de conexión**
        
No se pudo conectar con la API del backend.

**Detalles técnicos:** {str(e)}

**Verifica que:**
1. El contenedor backend esté corriendo
2. La API esté disponible en http://backend:8000
"""

# ======================================================
# 2. Estado del sistema
# ======================================================
def obtener_estado():
    """Obtiene el estado del sistema"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            r = response.json()
            status_icon = "[OK]" if r['status'] == 'healthy' else "[ERROR]"
            modelo_icon = "[OK]" if r['modelo_cargado'] else "[ERROR]"
            datos_icon = "[OK]" if r['datos_cargados'] else "[ERROR]"
            return f"""{status_icon} **Estado del sistema:** {r['status']}
{modelo_icon} **Modelo cargado:** {'Sí' if r['modelo_cargado'] else 'No'}
{datos_icon} **Datos cargados:** {'Sí' if r['datos_cargados'] else 'No'}
**Timestamp:** {r['timestamp']}"""
        else:
            return "Error obteniendo estado"
    except Exception as e:
        return f"Error: {str(e)}"

# ======================================================
# 3. Información del modelo
# ======================================================
def info_modelo():
    """Obtiene información del modelo"""
    try:
        response = requests.get(f"{API_URL}/model_info")
        if response.status_code == 200:
            r = response.json()
            params = r.get('parametros', {})
            return f"""**Información del Modelo**

**Tipo:** {r['tipo_modelo']}
**Ruta:** {r['modelo_path']}
**N° Features:** {r['n_features']}

**Parámetros:**
- Max Depth: {params.get('max_depth', 'N/A')}
- Min Samples Split: {params.get('min_samples_split', 'N/A')}
- Min Samples Leaf: {params.get('min_samples_leaf', 'N/A')}
- Random State: {params.get('random_state', 'N/A')}
"""
        else:
            return "Error obteniendo información del modelo"
    except Exception as e:
        return f"Error: {str(e)}"

# ======================================================
# 4. Interfaz Gradio
# ======================================================
with gr.Blocks(title="SodAI Drinks - Predicción de Compra", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SodAI Drinks - Sistema de Predicción de Compra")
    gr.Markdown("Sistema de Machine Learning para predecir la probabilidad de compra de productos por cliente.")
    
    with gr.Tabs():
        # --- Tab 1: Predicción
        with gr.TabItem("Predicción"):
            gr.Markdown("### Ingrese los datos para realizar una predicción")
            with gr.Row():
                cliente_id = gr.Number(label="ID Cliente", value=254403, precision=0)
                producto_id = gr.Number(label="ID Producto", value=34092, precision=0)
                semana = gr.Number(label="Semana (opcional)", value=None, precision=0)
            boton_predecir = gr.Button("Realizar Predicción", variant="primary")
            salida_prediccion = gr.Markdown("*Ingresa los datos y presiona 'Realizar Predicción'*")
            boton_predecir.click(fn=predecir, inputs=[cliente_id, producto_id, semana], outputs=salida_prediccion)

        # --- Tab 2: Estado del Sistema
        with gr.TabItem("Estado del Sistema"):
            boton_estado = gr.Button("Actualizar Estado", variant="secondary")
            salida_estado = gr.Markdown()
            boton_estado.click(fn=obtener_estado, outputs=salida_estado)
            demo.load(fn=obtener_estado, outputs=salida_estado)

        # --- Tab 3: Información del Modelo
        with gr.TabItem("Modelo"):
            boton_info = gr.Button("Ver Información", variant="secondary")
            salida_info = gr.Markdown()
            boton_info.click(fn=info_modelo, outputs=salida_info)
            demo.load(fn=info_modelo, outputs=salida_info)

    gr.Markdown("""
    ---
    **Tecnologías utilizadas:**
    - Backend: FastAPI  
    - Modelo: DecisionTreeClassifier (sklearn)  
    - Optimización: Optuna  
    - Tracking: MLflow  
    - Orquestación: Apache Airflow  

    **Proyecto:** SodAI Drinks - Entrega 2 | Laboratorio MDS 2025-2
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)