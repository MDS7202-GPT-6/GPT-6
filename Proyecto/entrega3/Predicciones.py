#!/usr/bin/env python3
"""Predicciones - versión actualizada

Carga el mejor modelo y pipeline generados por la entrega2 (Airflow).
- Usa el metadata `best_model_metadata.json` en el directorio de modelos si existe.
- Predice para la próxima semana basada en `transacciones.parquet` (max Semana + 1).
- Alinea features por nombre cuando es posible (recomendado).

Uso:
    python Predicciones.py 

Opciones relevantes:
    --models-dir PATH    Directorio donde Airflow guarda modelos (por defecto entrega2/airflow/data/models)
    --raw-dir PATH       Directorio raw con transacciones/clientes/productos (por defecto entrega2/airflow/data/raw)
    --out PATH           CSV de salida
    --sample-customers N Muestra de clientes (por defecto 100)
    --sample-products N  Muestra de productos (por defecto 50)
    --prob-threshold f   Umbral de probabilidad (por defecto 0.5)

"""
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
import json
import sys


def load_metadata(models_dir: Path):
    meta_path = models_dir / 'best_model_metadata.json'
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return None
    return None


def find_latest_model(models_dir: Path):
    files = list(models_dir.glob('*.pkl'))
    if not files:
        raise FileNotFoundError(f'No se encontraron modelos en {models_dir}')
    # prefer model files that match pattern *_modelo_*.pkl
    candidates = [p for p in files if 'modelo' in p.name]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return max(files, key=lambda p: p.stat().st_mtime)


def safe_joblib_load(p: Path):
    if not p.exists():
        raise FileNotFoundError(f'Archivo no encontrado: {p}')
    return joblib.load(p)


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
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', default=str(repo_root / 'Proyecto' / 'entrega2' / 'airflow' / 'data' / 'models'))
    parser.add_argument('--raw-dir', default=str(repo_root / 'Proyecto' / 'entrega2' / 'airflow' / 'data' / 'raw'))
    parser.add_argument('--out', default=str(repo_root / 'Proyecto' / 'entrega3' / 'predicciones_next_week.csv'))
    parser.add_argument('--sample-customers', type=int, default=100, help='Muestra de customers para predecir (por defecto 100). Use --use-all to predict for todos los IDs de transacciones.')
    parser.add_argument('--sample-products', type=int, default=50, help='Muestra de products para predecir (por defecto 50). Use --use-all to predict para todos los IDs de transacciones.')
    parser.add_argument('--use-all', action='store_true', help='Usar todos los customer_id y product_id presentes en transacciones.parquet (sin muestreo)')
    parser.add_argument('--prob-threshold', type=float, default=0.5)
    parser.add_argument('--batch', type=int, choices=[1,2,3,4], help='Número de batch según enunciado (1..4). Si se pasa, anula inferencia por raw.')
    parser.add_argument('--start', type=str, help='Fecha de inicio objetivo YYYY-MM-DD (anula batch y comportamiento por defecto)')
    parser.add_argument('--end', type=str, help='Fecha fin objetivo YYYY-MM-DD (anula batch y comportamiento por defecto)')
    parser.add_argument('--chunk-size', type=int, default=200, help='Tamaño de chunk (nº clientes por batch)')
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)

    # Informar rutas usadas
    print('CONFIG:')
    print('  raw_dir   =', raw_dir)
    print('  models_dir=', models_dir)
    print('  out_path  =', out_path)
    print('  sample_customers=', args.sample_customers, ' sample_products=', args.sample_products, ' prob_threshold=', args.prob_threshold)

    # Cargar raw data
    trans_path = raw_dir / 'transacciones.parquet'
    clientes_path = raw_dir / 'clientes.parquet'
    productos_path = raw_dir / 'productos.parquet'

    if not trans_path.exists():
        print(f'ERROR: no se encontró {trans_path}', file=sys.stderr)
        sys.exit(2)

    df_trans = pd.read_parquet(trans_path)
    df_trans['purchase_date'] = pd.to_datetime(df_trans['purchase_date'])
    print(f'DATA: transacciones rows={len(df_trans)}')

    # Determinar la última semana y año presentes y predecir para la siguiente
    df_trans['Semana'] = df_trans['purchase_date'].dt.isocalendar().week
    df_trans['Año'] = df_trans['purchase_date'].dt.year
    max_sem = int(df_trans['Semana'].max())
    max_year = int(df_trans['Año'].max())

    # Si el usuario especifica un batch o un rango de fechas, lo respetamos.
    target_start = None
    target_end = None
    if args.batch:
        # Mapeo de batches según el enunciado (fechas incluyentes)
        batch_map = {
            1: ('2025-01-01', '2025-01-05'),
            2: ('2025-01-06', '2025-01-12'),
            3: ('2025-01-13', '2025-01-19'),
            4: ('2025-01-20', '2025-01-26'),
        }
        target_start = pd.to_datetime(batch_map[args.batch][0])
        target_end = pd.to_datetime(batch_map[args.batch][1])
        print(f'Batch {args.batch} especificado: periodo objetivo {target_start.date()} - {target_end.date()}')
    elif args.start or args.end:
        if not (args.start and args.end):
            print('ERROR: --start y --end deben proporcionarse juntos cuando se usan.', file=sys.stderr)
            sys.exit(2)
        target_start = pd.to_datetime(args.start)
        target_end = pd.to_datetime(args.end)
        print(f'Rango de fechas especificado: {target_start.date()} - {target_end.date()}')
    else:
        # Comportamiento por defecto: semana siguiente al max presente en raw
        target_sem = max_sem + 1
        target_year = max_year
        # intentar derivar fecha de inicio/fin desde ISO week
        try:
            from datetime import date
            # ISO weekday 1 = Monday
            target_start = date.fromisocalendar(target_year, target_sem, 1)
            target_end = date.fromisocalendar(target_year, target_sem, 7)
        except Exception:
            target_start = None
            target_end = None
        print(f'Última semana en raw: semana={max_sem}, año={max_year}; prediciendo para Semana={target_sem}, Año={target_year}')

    # Si tenemos target_start/target_end, setear Semana/Año acorde al target_start
    if target_start is not None:
        # asegurar timestamps
        target_start = pd.to_datetime(target_start)
        target_end = pd.to_datetime(target_end)
        iso = target_start.isocalendar()
        target_sem = int(iso.week)
        target_year = int(iso.year)
        print(f'Periodo objetivo usado: {target_start.date()} - {target_end.date()} -> Semana={target_sem}, Año={target_year}')

    # Cargar clientes y productos si existen
    if clientes_path.exists():
        df_clients = pd.read_parquet(clientes_path)
        print(f'DATA: clientes rows={len(df_clients)}')
    else:
        df_clients = pd.DataFrame({'customer_id': df_trans['customer_id'].unique()})
        print('DATA: clientes file not found, using unique customers from transacciones')

    if productos_path.exists():
        df_products = pd.read_parquet(productos_path)
        print(f'DATA: productos rows={len(df_products)}')
    else:
        df_products = pd.DataFrame({'product_id': df_trans['product_id'].unique()})
        print('DATA: productos file not found, using unique products from transacciones')

    # Determinar sample customers/products
    if args.use_all:
        # Obtener todos los ids únicos desde transacciones (garantiza cobertura completa)
        all_customers = pd.Series(df_trans['customer_id'].unique())
        all_products = pd.Series(df_trans['product_id'].unique())
        sample_customers = all_customers.sort_values().tolist()
        sample_products = all_products.sort_values().tolist()
        print(f'Usando TODOS los customer_id/product_id presentes en transacciones: {len(sample_customers)} customers x {len(sample_products)} products = {len(sample_customers)*len(sample_products)} combinaciones')
    else:
        sample_customers = df_clients['customer_id'].sample(min(args.sample_customers, len(df_clients)), random_state=42).tolist()
        sample_products = df_products['product_id'].sample(min(args.sample_products, len(df_products)), random_state=42).tolist()
        print(f'Se predecirán {len(sample_customers)} clientes x {len(sample_products)} productos = {len(sample_customers)*len(sample_products)} combinaciones (muestreo)')

    # Cargar metadata si existe
    metadata = load_metadata(models_dir)
    model_path = None
    pipeline_path = None

    if metadata and 'model_path' in metadata and 'pipeline_path' in metadata:
        # Resolver rutas absolutas si fuera necesario
        candidate_model = Path(metadata['model_path'])
        candidate_pipeline = Path(metadata['pipeline_path'])
        if not candidate_model.exists():
            # intentar en models_dir
            candidate_model = models_dir / Path(metadata['model_path']).name
        if not candidate_pipeline.exists():
            candidate_pipeline = models_dir / Path(metadata['pipeline_path']).name

        if candidate_model.exists():
            model_path = candidate_model
        if candidate_pipeline.exists():
            pipeline_path = candidate_pipeline

    # Fallback: buscar último modelo y pipeline estándar
    if model_path is None:
        model_path = find_latest_model(models_dir)
        print('Metadata no encontrada o incompleta; usando último modelo:', model_path.name)
    else:
        print('Usando modelo desde metadata:', model_path.name)

    # Mostrar metadata completa si existe
    if metadata:
        print('Metadata encontrada:')
        for k, v in metadata.items():
            print(' ', k, ':', v)

    if pipeline_path is None:
        # probar pipeline_pp_best_{ts}.pkl por timestamp en model filename
        import re
        m = re.search(r'_(\d{8}_\d{6})', model_path.name)
        if m:
            candidate = models_dir / f'pipeline_pp_best_{m.group(1)}.pkl'
            if candidate.exists():
                pipeline_path = candidate
        if pipeline_path is None:
            # fallback a pipeline_pp.pkl
            candidate = models_dir / 'pipeline_pp.pkl'
            if candidate.exists():
                pipeline_path = candidate

    if pipeline_path is None:
        raise FileNotFoundError('No se pudo localizar pipeline de preprocesamiento en models_dir')

    print('Cargando pipeline desde:', pipeline_path)
    pipeline = safe_joblib_load(pipeline_path)
    print('  pipeline type:', type(pipeline))
    if hasattr(pipeline, 'named_steps'):
        try:
            print('  pipeline.named_steps:', list(pipeline.named_steps.keys()))
        except Exception:
            pass

    print('Cargando modelo desde:', model_path)
    model = safe_joblib_load(model_path)
    print('  modelo tipo:', type(model))
    try:
        params = model.get_params()
        short = {k: params[k] for k in list(params)[:8]}
        print('  modelo.get_params() sample:', short)
    except Exception:
        pass
    if hasattr(model, 'n_features_in_'):
        try:
            print('  modelo.n_features_in_ =', model.n_features_in_)
        except Exception:
            pass

    # Construir df_pred_input (combinaciones)
    from itertools import product
    combos = list(product(sample_customers, sample_products))
    df_pred_input = pd.DataFrame(combos, columns=['customer_id', 'product_id'])
    df_pred_input['Semana'] = target_sem
    df_pred_input['Año'] = target_year

    # Merge con info si está disponible
    if not df_clients.empty:
        df_pred_input = df_pred_input.merge(df_clients, on='customer_id', how='left')
    if not df_products.empty:
        df_pred_input = df_pred_input.merge(df_products, on='product_id', how='left')

    # Transformar usando pipeline
    print('Transformando datos con pipeline...')
    try:
        X_trans = pipeline.transform(df_pred_input)
    except Exception as e:
        print('Error al transformar con pipeline:', e, file=sys.stderr)
        # intentar transformar por partes (feature aggregator step can require extra args)
        # Como fallback, intentar usar pipeline.transform on a minimal copy
        try:
            X_trans = pipeline.transform(df_pred_input.fillna(0))
        except Exception as e2:
            print('Fallo fallback transform:', e2, file=sys.stderr)
            raise

    # Intentar alinear por nombres de features si es posible
    model_n = getattr(model, 'n_features_in_', None)
    aligned = False
    if hasattr(pipeline, 'named_steps') and 'preprocessing' in pipeline.named_steps:
        preproc = pipeline.named_steps['preprocessing']
        try:
            input_cols = df_pred_input.columns.tolist()
            feature_names = preproc.get_feature_names_out(input_cols)
            # convertir a lista de strings
            feature_names = [str(f) for f in feature_names]
            X_df = pd.DataFrame(X_trans, columns=feature_names)
            # si metadata tiene feature_names, alineamos
            if metadata and metadata.get('feature_names'):
                target_features = metadata['feature_names']
                # rellenar columnas faltantes con 0 y recortar extras
                for f in target_features:
                    if f not in X_df.columns:
                        X_df[f] = 0
                # mantener solo en orden target_features
                X_df = X_df.reindex(columns=target_features, fill_value=0)
                X_trans_aligned = X_df.values
                aligned = True
                print(f'Alineado por feature names: {len(target_features)} features')
            else:
                # si no hay metadata pero las dimensiones coinciden, podemos usar
                if model_n is not None and X_df.shape[1] == model_n:
                    X_trans_aligned = X_df.values
                    aligned = True
                else:
                    # no tenemos metadata ni coincidencia dimensional
                    X_trans_aligned = X_trans
        except Exception as e:
            print('No fue posible extraer feature names del preprocessor:', e)
            X_trans_aligned = X_trans
    else:
        X_trans_aligned = X_trans

    # Si no alineado y modelo espera n features, intentar adaptar igual que en el DAG
    if not aligned and model_n is not None:
        if X_trans_aligned.shape[1] != model_n:
            diff = X_trans_aligned.shape[1] - model_n
            if abs(diff) <= 5:
                print(f'Adaptando features por diferencia pequeña ({diff})')
                if diff > 0:
                    X_trans_aligned = X_trans_aligned[:, :model_n]
                else:
                    pad = np.zeros((X_trans_aligned.shape[0], -diff))
                    X_trans_aligned = np.hstack([X_trans_aligned, pad])
            else:
                raise ValueError(f'Imposible alinear features: modelo espera={model_n}, pipeline produjo={X_trans_aligned.shape[1]}')

    # Predecir
    print('Generando predicciones...')
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_trans_aligned)
        if probs.shape[1] == 1:
            preds = model.predict(X_trans_aligned)
            positive_mask = np.array(preds) == 1
            prob_scores = np.array(preds).astype(float)
        else:
            prob_scores = probs[:, 1]
            positive_mask = prob_scores >= args.prob_threshold
    else:
        preds = model.predict(X_trans_aligned)
        try:
            positive_mask = np.array(preds) >= 0.5
            prob_scores = np.array(preds).astype(float)
        except Exception:
            positive_mask = np.array(preds) == 1
            prob_scores = np.array(preds).astype(float)

    df_out = df_pred_input.loc[positive_mask, ['customer_id', 'product_id']].copy()
    if df_out.empty:
        print('No se encontraron pares con probabilidad >=', args.prob_threshold)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=['customer_id', 'product_id']).to_csv(out_path, index=False)
        return

    df_out = df_out.drop_duplicates().reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f'Resultados: {len(df_out)} pares escritos en {out_path}')


if __name__ == '__main__':
    main()
