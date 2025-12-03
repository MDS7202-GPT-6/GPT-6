# Entrega3 - Contenedor para predicciones

Este contenedor reproduce el entorno de `entrega2` (mismas versiones de librerías)
para evitar problemas de deserialización (pickles) entre versiones de Python / scikit-learn / joblib.

Build:

```bash
docker build -t entrega3-predictor:latest Proyecto/entrega3
```

Ejecución (montar los directorios de `models` y `raw` desde `entrega2`):

```bash
docker run --rm \
  -v /full/path/to/Proyecto/entrega2/airflow/data/models:/data/models:ro \
  -v /full/path/to/Proyecto/entrega2/airflow/data/raw:/data/raw:ro \
  -v /full/path/to/Proyecto/entrega3/predictions:/app/Proyecto/entrega3/predictions \
  entrega3-predictor:latest
```

Notas:
- Si prefieres pasar rutas personalizadas para `PIPELINE_PATH`, `MODELS_DIR`, `RAW_DIR` o `OUT_DIR`, exporta variables de entorno o pásalas con `-e` al `docker run`.
- Este contenedor usa `python:3.10-slim` para reproducir el entorno y evitar errores como "TypeError: code() argument 13 must be str, not int".
