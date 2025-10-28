import gradio as gr
import requests

# ====================================================
# Configuración
# ====================================================

BACKEND_URL = "http://backend:8002"

# ====================================================
# Función principal del chat
# ====================================================

def chat_with_bot(message, history):
    """Envía el mensaje al backend y retorna la respuesta"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": message, "reset_conversation": False},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["response"]
        else:
            return f"❌ Error del servidor: {response.status_code}\n{response.text}"
            
    except requests.exceptions.Timeout:
        return "⏱️ Timeout: El servidor tardó demasiado en responder."
    except requests.exceptions.ConnectionError:
        return "🔌 Error: No se puede conectar con el backend."
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}"


# ====================================================
# Interfaz de Gradio
# ====================================================

ejemplos = [
    "¿Cuántos clientes hay en el dataset?",
    "¿Cuántos productos únicos existen?",
    "¿Cuál es el producto más vendido?",
    "¿Qué categorías de productos hay?",
    "Dame información del cliente 1000",
    "¿Cuántas transacciones ha hecho el cliente 500?",
    "¿Cuál es la categoría más vendida?",
    "¿Cuántas marcas diferentes hay?",
]

demo = gr.ChatInterface(
    fn=chat_with_bot,
    examples=ejemplos,
    title="🤖 SodAI Drinks - Chatbot Conversacional",
    description="""
    ### 👋 ¡Bienvenido!
    
    Puedo ayudarte a explorar los datos de:
    - 🧑‍🤝‍🧑 **Clientes** (tipos, regiones, comportamiento)
    - 📦 **Productos** (categorías, marcas, inventario)
    - 💳 **Transacciones** (ventas, tendencias, análisis)
    
    💡 **Tip:** Haz preguntas específicas para mejores resultados.
    
    **Modelo:** Groq (llama-3.1-8b-instant)
    """,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False
    )
