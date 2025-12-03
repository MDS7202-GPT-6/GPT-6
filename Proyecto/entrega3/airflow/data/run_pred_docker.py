#!/usr/bin/env python3
"""Runner para ejecutar Predicciones dentro del contenedor Airflow.
Usa rutas dentro del contenedor: /opt/airflow/data/...
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
import sys


def find_latest_model(models_dir: Path):
    files = list(models_dir.glob('modelo_*.pkl'))
    if not files:
        raise FileNotFoundError(f'No se encontraron modelos en {models_dir}')
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest


def load_pipeline(pipeline_path: Path):
    if not pipeline_path.exists():
        raise FileNotFoundError(f'pipeline_pp.pkl no encontrado en {pipeline_path}')
    return joblib.load(pipeline_path)


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f'Modelo no encontrado en {model_path}')
    return joblib.load(model_path)


def detect_date_column(df: pd.DataFrame):
    candidates = ['purchase_date', 'date', 'ds', 'fecha', 'timestamp']
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    raise ValueError('No se encontró columna de fecha en el DataFrame')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2025-01-01')
    parser.add_argument('--end', default='2025-01-05')
    parser.add_argument('--transacciones', default='/opt/airflow/data/raw/transacciones.parquet')
    parser.add_argument('--models-dir', default='/opt/airflow/data/models')
    parser.add_argument('--pipeline', default='/opt/airflow/data/models/pipeline_pp.pkl')
    parser.add_argument('--out', default='/opt/airflow/data/predictions/predicciones_2025-01-01_2025-01-05.csv')
    parser.add_argument('--prob-threshold', type=float, default=0.5)
    args = parser.parse_args()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    trans_path = Path(args.transacciones)
    models_dir = Path(args.models_dir)
    pipeline_path = Path(args.pipeline)
    out_path = Path(args.out)

    print('Cargando transacciones...')
    df = pd.read_parquet(trans_path)
    date_col = detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col])

    # construir candidato: cartesian customers x products x days
    customers = df['customer_id'].unique()
    products = df['product_id'].unique()
    dates = pd.date_range(start=start, end=end, freq='D')
    rows = []
    for d in dates:
        c = pd.Series(customers, name='customer_id')
        p = pd.Series(products, name='product_id')
        grid = pd.MultiIndex.from_product([c, p], names=['customer_id', 'product_id']).to_frame(index=False)
        grid['purchase_date'] = d
        rows.append(grid)
    candidate = pd.concat(rows, ignore_index=True)
    print('Candidate rows:', len(candidate))

    model_file = find_latest_model(models_dir)
    print('Modelo usado:', model_file.name)

    print('Cargando pipeline...')
    pipeline = load_pipeline(Path(pipeline_path))
    print('Cargando modelo...')
    model = load_model(model_file)

    X_input = candidate.copy()
    # Asegurar columnas temporales que espera el transformer
    X_input['purchase_date'] = pd.to_datetime(X_input['purchase_date'])
    X_input['Semana'] = X_input['purchase_date'].dt.isocalendar().week
    X_input['Año'] = X_input['purchase_date'].dt.year
    X_input['trimestre'] = X_input['purchase_date'].dt.quarter

    # Asegurar columnas numéricas y categóricas esperadas por el pipeline
    numeric_features = ["num_deliver_per_week", "num_visit_per_week", "size", "frecuencia", "frecuencia_categoria", "frecuencia_brand"]
    categorical_features = ["customer_type", "brand", "category", "segment", "package", "geo_cluster", "trimestre"]

    for col in numeric_features:
        if col not in X_input.columns:
            X_input[col] = 0
    for col in categorical_features:
        if col not in X_input.columns:
            # geo_cluster es entero, usar -1 como valor por defecto
            if col == 'geo_cluster':
                X_input[col] = -1
            else:
                X_input[col] = ''
    # Llamar al pipeline
    if hasattr(pipeline, 'transform'):
        X = pipeline.transform(X_input)
    else:
        X = X_input

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        positive_mask = probs[:, 1] >= args.prob_threshold
    else:
        preds = model.predict(X)
        positive_mask = np.array(preds) == 1

    df_pos = candidate.loc[positive_mask]
    result = df_pos[['customer_id', 'product_id']].drop_duplicates().reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print('Predicciones guardadas en', out_path, 'pares:', len(result))


if __name__ == '__main__':
    main()
