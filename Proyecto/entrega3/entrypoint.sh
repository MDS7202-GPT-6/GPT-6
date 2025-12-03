#!/usr/bin/env bash
set -euo pipefail

# Variables (se pueden sobreescribir con -e en docker run)
MODELS_DIR=${MODELS_DIR:-/data/models}
RAW_DIR=${RAW_DIR:-/data/raw}
PIPELINE_PATH=${PIPELINE_PATH:-/app/Proyecto/entrega2/airflow/data/models/pipeline_pp.pkl}
OUT_DIR=${OUT_DIR:-/app/Proyecto/entrega3/predictions}

mkdir -p "$OUT_DIR"

echo "Usando MODELS_DIR=$MODELS_DIR"
echo "Usando RAW_DIR=$RAW_DIR"
echo "Usando PIPELINE_PATH=$PIPELINE_PATH"
echo "Salida en $OUT_DIR"

# Ejecutar script de predicciones con los paths apropiados
python /app/Proyecto/entrega3/Predicciones.py \
  --raw-dir "$RAW_DIR" \
  --models-dir "$MODELS_DIR" \
  --out "$OUT_DIR/predicciones_next_week.csv" "$@"
