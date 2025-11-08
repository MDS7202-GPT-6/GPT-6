import gradio as gr
import requests

API_URL = "http://backend:8001"

# ======================================================
# Función de recomendaciones
# ======================================================
def recomendar(cliente_id, semana, top_n):
    """Genera recomendaciones de productos para un cliente"""
    data = {
        "cliente_id": int(cliente_id),
        "semana": int(semana) if semana else None,
        "top_n": int(top_n)
    }
    def _extract_name_from_obj(obj):
        if not obj or not isinstance(obj, dict):
            return None
        for k in ("nombre", "name", "product_name", "productName", "nombre_producto"):
            if k in obj and obj[k]:
                return obj[k]
        return None

    def _fetch_product_name_from_api(prod_id):
        try:
            resp = requests.get(f"{API_URL}/products/{int(prod_id)}", timeout=3)
            if resp.status_code == 200:
                pj = resp.json()
                name = _extract_name_from_obj(pj)
                if name:
                    return name
                # fallback: try keys that contain 'name' or 'nombre'
                for k, v in pj.items():
                    if isinstance(k, str) and ("name" in k.lower() or "nombre" in k.lower()):
                        return v
        except Exception:
            pass
        return None

    try:
        response = requests.post(f"{API_URL}/recommend", json=data)
        if response.status_code == 200:
            r = response.json()
            recomendaciones_md = []
            for rec in r.get('recomendaciones', []):
                prod_id = rec.get('producto_id')
                # intentar extraer nombre de la propia recomendación
                prod_name = _extract_name_from_obj(rec)
                # si no viene, consultar endpoint de producto
                if not prod_name and prod_id is not None:
                    prod_name = _fetch_product_name_from_api(prod_id)

                if not prod_name:
                    prod_name = f"Producto #{prod_id}"

                prob = rec.get('probabilidad_compra', 0.0)
                if prob >= 0.7:
                    nivel = "MUY ALTA"
                elif prob >= 0.5:
                    nivel = "ALTA"
                elif prob >= 0.3:
                    nivel = "MEDIA"
                else:
                    nivel = "BAJA"

                recomendaciones_md.append(
                    f"### {rec.get('ranking','?')}. **{prod_name} (ID: {prod_id})** - {prob:.1%} ({nivel})\n"
                    f"- **Categoría:** {rec.get('categoria','-')}\n"
                    f"- **Marca:** {rec.get('marca','-')}\n"
                    f"- **Sub-categoría:** {rec.get('sub_categoria','-')}\n"
                    f"- **Segmento:** {rec.get('segmento','-')}\n"
                )

            result = f"""# **RECOMENDACIONES DE PRODUCTOS**

---

## Cliente ID: **{r.get('cliente_id','-')}**

**Semana:** {r.get('semana','-')} ({r.get('año','-')}) | **Productos evaluados:** {r.get('n_productos_evaluados','-')}

### Información del Cliente
- **Tipo:** {r.get('cliente_info',{}).get('tipo','-')}
- **Región:** {r.get('cliente_info',{}).get('region','-')}
- **Zona:** {r.get('cliente_info',{}).get('zona','-')}

---

## Top {r.get('top_n','-')} Productos Recomendados

{"".join(recomendaciones_md)}

---

*Timestamp: {r.get('timestamp','-')}*

**Interpretación:** Estos son los productos con mayor probabilidad de ser comprados por este cliente en la semana {r.get('semana','-')}.
"""
            return result
        elif response.status_code == 404:
            error_detail = response.json().get('detail', 'Error desconocido')
            return f"""**Error: Cliente no encontrado**

{error_detail}

**Sugerencia:**
Verifica que el ID del cliente sea correcto.
"""
        else:
            try:
                error_detail = response.json().get('detail', 'Error desconocido')
            except Exception:
                error_detail = 'Error desconocido'
            return f"""**Error generando recomendaciones**

{error_detail}
"""
    except Exception as e:
        return f"""**Error de conexión**
        
No se pudo conectar con la API del backend.

**Detalles técnicos:** {str(e)}
"""

# ======================================================
# Estado del sistema
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
# Interfaz Gradio
# ======================================================
with gr.Blocks(title="SodAI Drinks - Sistema de Recomendación", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SodAI Drinks - Sistema de Recomendación de Productos")
    gr.Markdown("Genera recomendaciones personalizadas basadas en el historial de compra y probabilidad predictiva.")
    
    with gr.Tabs():
        # --- Tab 1: Recomendaciones
        with gr.TabItem("Recomendaciones"):
            gr.Markdown("### Sistema de Recomendación Personalizado")
            gr.Markdown("""
            Este sistema analiza el historial de compra de un cliente y evalúa **todos los productos disponibles** 
            para generar una lista de los productos con mayor probabilidad de compra.
            """)
            
            with gr.Row():
                cliente_id_rec = gr.Number(label="ID Cliente", value=254403, precision=0)
                semana_rec = gr.Number(label="Semana (opcional, deja vacío para próxima semana)", value=None, precision=0)
                top_n = gr.Slider(label="Top N productos a recomendar", minimum=1, maximum=20, value=5, step=1)
            
            boton_recomendar = gr.Button("Generar Recomendaciones", variant="primary", size="lg")
            salida_recomendaciones = gr.Markdown("*Ingresa el ID del cliente y presiona 'Generar Recomendaciones'*")
            
            boton_recomendar.click(fn=recomendar, inputs=[cliente_id_rec, semana_rec, top_n], outputs=salida_recomendaciones)
            
            with gr.Accordion("Cómo funciona el sistema de recomendación", open=False):
                gr.Markdown("""
                ### Metodología
                
                1. **Análisis de perfil**: El sistema identifica el perfil del cliente (tipo, región, zona)
                2. **Evaluación masiva**: Se evalúan todos los productos disponibles (~971 productos)
                3. **Scoring predictivo**: Cada producto recibe una probabilidad de compra basada en:
                   - Historial de compra del cliente
                   - Frecuencia de compra por producto, marca y categoría
                   - Características del producto
                   - Patrones temporales
                4. **Ranking**: Los productos se ordenan por probabilidad y se devuelven los top N
                
                ### Clasificación de probabilidades
                
                - **MUY ALTA**: 70-100% - Productos con alta afinidad demostrada
                - **ALTA**: 50-70% - Productos probables según patrón de compra
                - **MEDIA**: 30-50% - Productos con potencial moderado
                - **BAJA**: 0-30% - Productos menos probables
                
                ### Casos de uso
                
                - **Campañas de marketing**: Personalizar ofertas por cliente
                - **App móvil**: Mostrar productos recomendados al abrir la app
                - **Optimización de inventario**: Priorizar stock según probabilidad de venta
                - **Email marketing**: Enviar recomendaciones personalizadas
                """)

        # --- Tab 2: Estado del Sistema
        with gr.TabItem("Estado del Sistema"):
            gr.Markdown("### Monitoreo del sistema de recomendación")
            boton_estado = gr.Button("Actualizar Estado", variant="secondary")
            salida_estado = gr.Markdown()
            boton_estado.click(fn=obtener_estado, outputs=salida_estado)
            demo.load(fn=obtener_estado, outputs=salida_estado)

    gr.Markdown("""
    ---
    **Tecnologías utilizadas:**
    - Backend: FastAPI  
    - Frontend: Gradio  
    - Modelo: DecisionTreeClassifier (sklearn)  
    - Tracking: MLflow  
    - Orquestación: Apache Airflow  

    **Proyecto:** SodAI Drinks - Sistema de Recomendación | Laboratorio MDS 2025-2
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
