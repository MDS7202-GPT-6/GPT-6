# Entrega 3 — Orquestación y predicción (resumen)

Este repositorio contiene la entrega 3 del proyecto: un pipeline orquestado con Apache Airflow que realiza ingesta, preprocesado, (re)entrenamiento y generación de predicciones. En esta entrega se utiliza la copia del pipeline llamada "pipeline copy" y su identificador en Airflow es `pipeline_modelo_previo`.

Importante: para evitar confusiones solo debe levantarse el `docker-compose.yml` que está dentro de `entrega3/airflow/`.

Contenido relevante de `entrega3/`
--------------------------------
- `airflow/` : DAGs, configuración y `docker-compose.yml` para ejecutar Airflow.
- `data/models/` : modelos guardados (.pkl) y el pipeline de preprocesado (`pipeline_pp.pkl`).
- `predictions/` : resultados (predicciones) generadas por modelos que resultaron ser incorrectos (explicado en enunciado_entrega3.ipynb).
- `predictions-Previo/` : predicciones generadas por el modelo previo (el `pipeline_modelo_previo`) y que muestran resultados más correctos; usar como referencia para evaluación.

Cómo levantar el entorno Docker (usar solo el compose dentro de `airflow/`)
--------------------------------------------------------------------
1. Sitúate en la carpeta del compose de Airflow:

```bash
cd Proyecto/entrega3/airflow
```

2. Construir (si es necesario) y levantar los servicios en segundo plano:

```bash
docker compose build
docker compose up -d
```

3. Verificar que los servicios estén arriba:

```bash
docker compose ps
docker compose logs -f
```

4. UIs comunes:
- Airflow: http://localhost:8080


Pipeline utilizado
------------------
- Nombre en el repositorio: copia "pipeline copy".
- DAG en Airflow: `pipeline_modelo_previo` (es la que debe ejecutarse para reproducir los resultados del modelo previo).

Explicación de `predictions/` vs `predictions-Previo/`
----------------------------------------------------
- `predictions/` contiene los resultados (archivos de salida) generados por versiones del/los modelos que terminaron siendo erróneos o con peor desempeño. Sirven para análisis de fallos y debugging.
- `predictions-Previo/` contiene las salidas del modelo entrenado previamente (el que corresponde a `pipeline_modelo_previo`) y que presenta resultados más correctos; esta carpeta es la referencia para evaluación y comparaciones.
