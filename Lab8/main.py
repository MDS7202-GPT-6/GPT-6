from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import pickle
import traceback
import uvicorn


app = FastAPI(
    title="API de Potabilidad del Agua",
    description="Predice si el agua es potable o no según sus características químicas.",
    version="2.0"
)


try:
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Modelo cargado correctamente.")
except Exception as e:
    print("No se pudo cargar el modelo:", e)
    model = None


# Nota: Se creó una míni vista web para facilitar la interacción con la API.
# ver en http://127.0.0.1:8000
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <html>
    <head>
        <title>💧 API de Potabilidad del Agua</title>
        <style>
            body { font-family: Arial; margin: 50px; background: #f7f9fc; color: #333; }
            h1 { color: #0077b6; }
            form { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 420px; }
            label { display: block; margin-top: 10px; font-weight: bold; }
            input { width: 100%; padding: 8px; margin-top: 4px; border-radius: 6px; border: 1px solid #ccc; }
            button { margin-top: 15px; background-color: #0077b6; color: white; padding: 10px; border: none; border-radius: 6px; cursor: pointer; }
            button:hover { background-color: #0096c7; }
            .result { margin-top: 20px; font-weight: bold; font-size: 1.2em; }
        </style>
    </head>
    <body>
        <h1>💧 API de Potabilidad del Agua</h1>
        <p>Ingrese las mediciones químicas para predecir si el agua es potable.</p>

        <form action="/prediccion" method="post">
            <label>pH:</label><input type="number" step="any" name="ph" required>
            <label>Hardness:</label><input type="number" step="any" name="Hardness" required>
            <label>Solids:</label><input type="number" step="any" name="Solids" required>
            <label>Chloramines:</label><input type="number" step="any" name="Chloramines" required>
            <label>Sulfate:</label><input type="number" step="any" name="Sulfate" required>
            <label>Conductivity:</label><input type="number" step="any" name="Conductivity" required>
            <label>Organic_carbon:</label><input type="number" step="any" name="Organic_carbon" required>
            <label>Trihalomethanes:</label><input type="number" step="any" name="Trihalomethanes" required>
            <label>Turbidity:</label><input type="number" step="any" name="Turbidity" required>
            <button type="submit">🔍 Predecir</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/prediccion", response_class=HTMLResponse)
def prediccion(
    ph: float = Form(...),
    Hardness: float = Form(...),
    Solids: float = Form(...),
    Chloramines: float = Form(...),
    Sulfate: float = Form(...),
    Conductivity: float = Form(...),
    Organic_carbon: float = Form(...),
    Trihalomethanes: float = Form(...),
    Turbidity: float = Form(...)
):
    try:
        df = pd.DataFrame([{
            "ph": ph,
            "Hardness": Hardness,
            "Solids": Solids,
            "Chloramines": Chloramines,
            "Sulfate": Sulfate,
            "Conductivity": Conductivity,
            "Organic_carbon": Organic_carbon,
            "Trihalomethanes": Trihalomethanes,
            "Turbidity": Turbidity
        }])

        pred = model.predict(df)
        potabilidad = int(pred[0])

        color = "#2a9d8f" if potabilidad == 1 else "#e63946"
        mensaje = "El agua es POTABLE" if potabilidad == 1 else "El agua NO es potable"

        html_resultado = f"""
        <html><body style="font-family:Arial; background:#f7f9fc; margin:50px;">
            <h1 style="color:#0077b6;">Resultado de Predicción</h1>
            <div style="background:white; padding:20px; border-radius:10px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.1); width:400px;">
                <p class="result" style="color:{color}; font-size:1.3em;">{mensaje}</p>
                <a href="/" style="text-decoration:none; color:#0077b6;">Volver</a>
            </div>
        </body></html>
        """
        return HTMLResponse(content=html_resultado)

    except Exception as e:
        print("Error interno:", e)
        print(traceback.format_exc())
        return HTMLResponse(f"<p>Error: {str(e)}</p>", status_code=500)



@app.post("/potabilidad/")
def predecir_json(muestra: dict):
    try:
        df = pd.DataFrame([muestra])
        pred = model.predict(df)
        potabilidad = int(pred[0])
        return {"potabilidad": potabilidad}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)