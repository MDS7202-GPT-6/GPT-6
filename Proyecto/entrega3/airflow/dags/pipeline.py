from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import os
import joblib
from helper_functions import (
    load_data,
    preprocess_data,
    detect_drift,
    train_and_log_model,
)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["alertas@mlpipeline.cl"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="pipeline_modelo_lightgbm",
    default_args=default_args,
    description="Pipeline de modelado DecisionTree + Optuna + MLflow + SHAP",
    schedule_interval="@weekly",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["mlops", "decisiontree", "optuna", "mlflow"],
) as dag:


    # ============================================================
    # 🟦 1) INGESTA
    # ============================================================
    def task_ingesta(**context):
        df_transacciones, df_clientes, df_productos = load_data()
        ti = context["ti"]
        ti.xcom_push(key="df_transacciones", value=df_transacciones.to_json())
        ti.xcom_push(key="df_clientes", value=df_clientes.to_json())
        ti.xcom_push(key="df_productos", value=df_productos.to_json())

    ingesta_task = PythonOperator(
        task_id="ingestar_datos",
        python_callable=task_ingesta,
        provide_context=True,
    )


    # ============================================================
    # 🟦 2) PREPROCESAMIENTO
    # ============================================================
    def task_preprocesar(**context):
        import pandas as pd
        import numpy as np

        ti = context["ti"]
        df_transacciones = pd.read_json(ti.xcom_pull(task_ids="ingestar_datos", key="df_transacciones"))
        df_clientes = pd.read_json(ti.xcom_pull(task_ids="ingestar_datos", key="df_clientes"))
        df_productos = pd.read_json(ti.xcom_pull(task_ids="ingestar_datos", key="df_productos"))
        
        X_train, X_val, y_train, y_val, X_test, y_test, pipeline_pp = preprocess_data(
            df_transacciones, df_clientes, df_productos
        )

        np.save('/tmp/X_train.npy', X_train)
        np.save('/tmp/X_val.npy', X_val)
        np.save('/tmp/y_train.npy', y_train)
        np.save('/tmp/y_val.npy', y_val)
        np.save('/tmp/X_test.npy', X_test)
        np.save('/tmp/y_test.npy', y_test)

        joblib.dump(pipeline_pp, '/tmp/pipeline_pp.pkl')
        joblib.dump(pipeline_pp, '/opt/airflow/data/models/pipeline_pp.pkl')

    preprocesar_task = PythonOperator(
        task_id="preprocesar_datos",
        python_callable=task_preprocesar,
        provide_context=True,
    )


    # ============================================================
    # 🟦 3) DETECTAR DRIFT
    # ============================================================
    def task_detectar_drift(**context):
        import pandas as pd
        import os

        ti = context["ti"]

        old_dir = "/opt/airflow/data/raw_old"
        new_dir = "/opt/airflow/data/raw"
        os.makedirs(old_dir, exist_ok=True)

        df_new = pd.read_json(ti.xcom_pull(task_ids="ingestar_datos", key="df_transacciones"))

        old_path = os.path.join(old_dir, "df_transacciones_old.parquet")
        if not os.path.exists(old_path):
            print("⚠️ Primera ejecución → reentrenar")
            return "reentrenar_modelo"

        df_old = pd.read_parquet(old_path)
        drift = detect_drift(
            df_old[["customer_id","product_id","purchase_date","items"]],
            df_new[["customer_id","product_id","purchase_date","items"]],
            threshold=0.1
        )

        if drift:
            return "reentrenar_modelo"
        else:
            return "usar_modelo_existente"

    detectar_drift_task = BranchPythonOperator(
        task_id="detectar_drift",
        python_callable=task_detectar_drift,
        provide_context=True,
    )


    # ============================================================
    # 🟦 4) REENTRENAMIENTO (+ REENTRENO FINAL CON TODOS LOS DATOS)
    # ============================================================
    def task_reentrenar(**context):
        import numpy as np
        import os
        import pandas as pd

        ti = context["ti"]

        X_train = np.load('/tmp/X_train.npy', allow_pickle=True)
        X_val = np.load('/tmp/X_val.npy', allow_pickle=True)
        y_train = np.load('/tmp/y_train.npy', allow_pickle=True)
        y_val = np.load('/tmp/y_val.npy', allow_pickle=True)
        X_test = np.load('/tmp/X_test.npy', allow_pickle=True)
        y_test = np.load('/tmp/y_test.npy', allow_pickle=True)

        model_dir = "/opt/airflow/data/models"
        modelos = [f for f in os.listdir(model_dir) if f.endswith(".pkl") and not f.startswith("pipeline")]
        
        optimize = len(modelos) == 0
        if optimize:
            print("🎯 Primera ejecución → Optuna")
        else:
            print("♻️ Reentrenamiento por drift → sin Optuna")

        model_final = train_and_log_model(X_train, X_val, y_train, y_val, X_test, y_test, optimize)

        final_path = "/opt/airflow/data/models/modelo_final.pkl"
        joblib.dump(model_final, final_path)
        ti.xcom_push(key="model_path", value=final_path)

        df_new = pd.read_json(ti.xcom_pull(task_ids="ingestar_datos", key="df_transacciones"))
        df_new.to_parquet("/opt/airflow/data/raw_old/df_transacciones_old.parquet")

    reentrenar_task = PythonOperator(
        task_id="reentrenar_modelo",
        python_callable=task_reentrenar,
        provide_context=True,
    )


    # ============================================================
    # 🟦 5) USAR MODELO EXISTENTE
    # ============================================================
    def task_usar_modelo_existente(**context):
        model_dir = "/opt/airflow/data/models"
        modelos = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pkl")]
        modelos = [m for m in modelos if "final" in m]

        if not modelos:
            raise FileNotFoundError("⚠️ No existe modelo_final.pkl")

        latest = max(modelos, key=os.path.getctime)
        context["ti"].xcom_push(key="model_path", value=latest)

    usar_modelo_existente_task = PythonOperator(
        task_id="usar_modelo_existente",
        python_callable=task_usar_modelo_existente,
        provide_context=True,
    )


    # ============================================================
    # 🟦 6) GENERAR PREDICCIONES A PARTIR DEL CSV REAL
    # ============================================================
    def task_predecir(**context):
        import numpy as np
        import pandas as pd
        import joblib
        import os

        ti = context["ti"]

        # Modelo final
        model_path = ti.xcom_pull(task_ids=["reentrenar_modelo", "usar_modelo_existente"], key="model_path")
        model_path = [m for m in model_path if m is not None][0]
        model = joblib.load(model_path)

        # Pipeline
        pipeline_pp = joblib.load('/opt/airflow/data/models/pipeline_pp.pkl')

        # Datasets originales
        df_transacciones = pd.read_json(ti.xcom_pull(task_ids="ingestar_datos", key="df_transacciones"))
        df_clientes = pd.read_json(ti.xcom_pull(task_ids="ingestar_datos", key="df_clientes"))
        df_productos = pd.read_json(ti.xcom_pull(task_ids="ingestar_datos", key="df_productos"))

        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])
        df_transacciones['Semana'] = df_transacciones['purchase_date'].dt.isocalendar().week
        df_transacciones['Año'] = df_transacciones['purchase_date'].dt.year

        next_week = df_transacciones['Semana'].max() + 1
        this_year = df_transacciones['Año'].max()

        # ============================================================
        #   🔥 CREAR future_pairs.csv AUTOMÁTICAMENTE SI NO EXISTE
        # ============================================================
        csv_path = "/opt/airflow/data/future/future_pairs.csv"

        if not os.path.exists(csv_path):
            print("⚠️ No existe future_pairs.csv → creando automáticamente...")

            # tomar última semana observada
            ultima_semana = df_transacciones["Semana"].max()
            ultima_year = df_transacciones["Año"].max()

            df_lastweek = df_transacciones[
                (df_transacciones["Semana"] == ultima_semana) &
                (df_transacciones["Año"] == ultima_year)
            ][["customer_id", "product_id"]].drop_duplicates()

            os.makedirs("/opt/airflow/data/future", exist_ok=True)
            df_lastweek.to_csv(csv_path, index=False)

            print(f"📄 Archivo creado automáticamente con {len(df_lastweek)} filas: {csv_path}")

        # Ahora ya existe, así que lo cargamos
        df_pred_input = pd.read_csv(csv_path)
        print(f"📄 Cargado CSV con {df_pred_input.shape[0]} pares para predecir")

        # Completar con semana y año futuro
        df_pred_input["Semana"] = next_week
        df_pred_input["Año"] = this_year

        # Merge metadata
        df_pred_input = df_pred_input.merge(df_clientes, on="customer_id", how="left")
        df_pred_input = df_pred_input.merge(df_productos, on="product_id", how="left")

        # Transformación
        X_pred = pipeline_pp.transform(df_pred_input)
        proba = model.predict_proba(X_pred)[:, 1]

        df_pred_input["probabilidad_compra"] = proba
        df_pred_input = df_pred_input.sort_values("probabilidad_compra", ascending=False)

        # Guardar
        out = f"/opt/airflow/data/predictions/pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_pred_input.to_csv(out, index=False)

        print(f"💾 Predicciones guardadas en: {out}")

    predecir_task = PythonOperator(
        task_id="generar_predicciones",
        python_callable=task_predecir,
        provide_context=True,
        trigger_rule="one_success"
    )


    fin_ok = EmptyOperator(task_id="fin_pipeline")


    (
        ingesta_task
        >> preprocesar_task
        >> detectar_drift_task
        >> [reentrenar_task, usar_modelo_existente_task]
        >> predecir_task
        >> fin_ok
    )