from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import os
import joblib
import numpy as np
import pandas as pd

from helper_functions_experiment import (
    load_data,
    preprocess_data,
    detect_drift,
    model_search,
    optimize_threshold,
)


# ============================================================
# CONFIGURACIÓN BASE
# ============================================================

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
    dag_id="pipeline_modelo_experiment-final",
    default_args=default_args,
    description="Pipeline experimental: agrega calibración, Optuna y monitoring",
    schedule_interval="@weekly",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["mlops", "experiment", "monitoring"],
) as dag:

    # ============================================================
    # TAREA 1: INGESTA
    # ============================================================
    def task_ingesta(**context):
        print("📥 Ingestando datos...")
        df_trans, df_cli, df_prod = load_data()
        ti = context["ti"]

        ti.xcom_push("df_transacciones", df_trans.to_json())
        ti.xcom_push("df_clientes", df_cli.to_json())
        ti.xcom_push("df_productos", df_prod.to_json())
        print("📥 Ingesta completa.")

    ingesta_task = PythonOperator(
        task_id="ingestar_datos",
        python_callable=task_ingesta,
        provide_context=True,
    )

    # ============================================================
    # TAREA 2: PREPROCESAMIENTO
    # ============================================================
    def task_preprocesar(**context):
        print("⚙️ Iniciando preprocesamiento...")
        ti = context["ti"]

        df_trans = pd.read_json(ti.xcom_pull(key="df_transacciones", task_ids="ingestar_datos"))
        df_cli = pd.read_json(ti.xcom_pull(key="df_clientes", task_ids="ingestar_datos"))
        df_prod = pd.read_json(ti.xcom_pull(key="df_productos", task_ids="ingestar_datos"))

        (
            X_train,
            X_val,
            y_train,
            y_val,
            X_test,
            y_test,
            pipeline_pp,
            X_all,
            y_all,
        ) = preprocess_data(df_trans, df_cli, df_prod)

        print("💾 Guardando matrices transformadas...")

        np.save("/tmp/X_train.npy", X_train)
        np.save("/tmp/X_val.npy", X_val)
        np.save("/tmp/y_train.npy", y_train)
        np.save("/tmp/y_val.npy", y_val)
        np.save("/tmp/X_test.npy", X_test)
        np.save("/tmp/y_test.npy", y_test)

        np.save("/tmp/X_all.npy", X_all)
        np.save("/tmp/y_all.npy", y_all)

        joblib.dump(pipeline_pp, "/tmp/pipeline_pp.pkl")
        joblib.dump(pipeline_pp, "/opt/airflow/data/models/pipeline_pp.pkl")

        print("⚙️ Preprocesamiento completado.")

    preprocesar_task = PythonOperator(
        task_id="preprocesar_datos",
        python_callable=task_preprocesar,
        provide_context=True,
    )

    # ============================================================
    # TAREA 3: DETECTAR DRIFT
    # ============================================================

    def task_detectar_drift(**context):
        print("🔍 Detectando drift...")
        ti = context["ti"]

        df_new = pd.read_json(ti.xcom_pull(key="df_transacciones", task_ids="ingestar_datos"))

        old_dir = "/opt/airflow/data/raw_old"
        os.makedirs(old_dir, exist_ok=True)

        old_path = f"{old_dir}/df_transacciones_old.parquet"

        # Si no existe histórico → entrenar
        if not os.path.exists(old_path):
            print("⚠️ No hay histórico old → reentrenar")
            return "reentrenar_modelo"

        df_old = pd.read_parquet(old_path)

        drift = detect_drift(
            df_old[["customer_id","product_id","purchase_date","items"]],
            df_new[["customer_id","product_id","purchase_date","items"]],
            threshold=0.1
        )

        if drift:
            print("⚠️ DRIFT DETECTADO → reentrenar")
            return "reentrenar_modelo"
        else:
            print("👌 No hay drift → usar modelo existente")
            return "usar_modelo_existente"

    detectar_drift_task = BranchPythonOperator(
        task_id="detectar_drift",
        python_callable=task_detectar_drift,
        provide_context=True,
    )

    # ============================================================
    # TAREA 4A: REENTRENAR MODELO (cuando hay drift)
    # ============================================================

    def task_reentrenar(**context):
        print("♻️ Reentrenando modelo por drift...")
        ti = context["ti"]

        X_train = np.load("/tmp/X_train.npy", allow_pickle=True)
        X_val   = np.load("/tmp/X_val.npy", allow_pickle=True)
        y_train = np.load("/tmp/y_train.npy", allow_pickle=True)
        y_val   = np.load("/tmp/y_val.npy", allow_pickle=True)
        X_test  = np.load("/tmp/X_test.npy", allow_pickle=True)
        y_test  = np.load("/tmp/y_test.npy", allow_pickle=True)

        print("🔍 Lanzando model_search (con Optuna)...")
        best_model, best_params, best_name, best_threshold = model_search(
            X_train, X_val, y_train, y_val,
            random_state=42,
            optuna_trials=40  # 💥 más experimentos
        )

        print(f"🏆 Mejor modelo: {best_name}")
        print(f"🧪 Params: {best_params}")
        print(f"🎯 Threshold óptimo: {best_threshold}")

        ti.xcom_push(key="best_model_name", value=best_name)
        ti.xcom_push(key="best_threshold", value=float(best_threshold))

        # ENTRENAMIENTO FINAL
        X_all = np.load("/tmp/X_all.npy", allow_pickle=True)
        y_all = np.load("/tmp/y_all.npy", allow_pickle=True)

        print("🔁 Reentrenando modelo FINAL con todo el dataset...")

        model_final = best_model.__class__(**best_params)
        model_final.fit(X_all, y_all)

        model_path = f"/opt/airflow/data/models/modelo_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(model_final, model_path)

        print(f"💾 Modelo final guardado en: {model_path}")

        # Actualizar histórico
        df_new = pd.read_json(ti.xcom_pull(key="df_transacciones", task_ids="ingestar_datos"))
        df_new.to_parquet("/opt/airflow/data/raw_old/df_transacciones_old.parquet")

        ti.xcom_push("model_path", model_path)

    reentrenar_task = PythonOperator(
        task_id="reentrenar_modelo",
        python_callable=task_reentrenar,
        provide_context=True,
    )

    # ============================================================
    # TAREA 4B: USAR MODELO EXISTENTE
    # ============================================================

    def task_usar_modelo_existente(**context):
        print("📦 Cargando modelo existente...")
        model_dir = "/opt/airflow/data/models"

        modelos = [os.path.join(model_dir, f)
                   for f in os.listdir(model_dir)
                   if f.endswith(".pkl") and not f.startswith("pipeline")]

        if not modelos:
            raise FileNotFoundError("❌ No hay modelos previos.")

        finales = [m for m in modelos if "modelo_final" in m]
        selected = max(finales, key=os.path.getctime) if finales else max(modelos, key=os.path.getctime)

        print(f"📦 Usando modelo: {selected}")
        context["ti"].xcom_push("model_path", selected)

    usar_modelo_existente_task = PythonOperator(
        task_id="usar_modelo_existente",
        python_callable=task_usar_modelo_existente,
        provide_context=True,
    )
    # ============================================================
    # TAREA 6: PREDICCIÓN (USA future_pairs O LO CREA)
    # ============================================================

    def task_predecir(**context):
        print("🔮 Generando predicciones...")
        ti = context["ti"]

                # ================================
        #  RESOLVER model_path (LazyXComAccess)
        # ================================
        model_path = ti.xcom_pull(
            task_ids=["reentrenar_modelo", "usar_modelo_existente"],
            key="model_path"
        )

        # Caso 1: viene como LazyXComAccess → resolver()
        try:
            if hasattr(model_path, "resolve"):  
                model_path = model_path.resolve(context=context)
        except Exception:
            pass

        # Caso 2: viene como lista → tomar el primero válido
        if isinstance(model_path, list):
            model_path = [m for m in model_path if m]  # filtrar None
            model_path = model_path[0] if model_path else None

        # Caso 3: si aún no es string → sacar directo desde cada task
        if not isinstance(model_path, (str, bytes, os.PathLike)):
            # Pull directo desde reentrenar
            mp1 = ti.xcom_pull(task_ids="reentrenar_modelo", key="model_path")
            # Pull directo desde usar_modelo_existente
            mp2 = ti.xcom_pull(task_ids="usar_modelo_existente", key="model_path")
            # Seleccionar el primero válido
            model_path = mp1 if isinstance(mp1, str) else mp2

        # Validación final
        if not isinstance(model_path, (str, bytes, os.PathLike)):
            raise FileNotFoundError("❌ model_path sigue inválido incluso tras resolver LazyXComAccess")

        print(f"📦 Usando modelo desde: {model_path}")
        model = joblib.load(model_path)
        pipeline_pp = joblib.load("/tmp/pipeline_pp.pkl")

        df_trans = pd.read_json(ti.xcom_pull(key="df_transacciones", task_ids="ingestar_datos"))
        df_cli   = pd.read_json(ti.xcom_pull(key="df_clientes", task_ids="ingestar_datos"))
        df_prod  = pd.read_json(ti.xcom_pull(key="df_productos", task_ids="ingestar_datos"))

        df_trans["purchase_date"] = pd.to_datetime(df_trans["purchase_date"])
        df_trans["Semana"] = df_trans["purchase_date"].dt.isocalendar().week
        df_trans["Año"] = df_trans["purchase_date"].dt.year

        next_week = df_trans["Semana"].max() + 1
        year_now = df_trans["Año"].max()

        # Crear archivo si no existe
        csv_path = "/opt/airflow/data/future/future_pairs.csv"
        os.makedirs("/opt/airflow/data/future", exist_ok=True)

        if not os.path.exists(csv_path):
            print("⚠️ No existe future_pairs.csv → generándolo automáticamente...")
            df_lastweek = df_trans[
                df_trans["Semana"] == df_trans["Semana"].max()
            ][["customer_id","product_id"]].drop_duplicates()
            df_lastweek.to_csv(csv_path, index=False)
            print(f"📄 future_pairs.csv creado con {len(df_lastweek)} filas.")

        df_pred_input = pd.read_csv(csv_path)
        df_pred_input["Semana"] = next_week
        df_pred_input["Año"] = year_now

        df_pred_input = df_pred_input.merge(df_cli, on="customer_id", how="left")
        df_pred_input = df_pred_input.merge(df_prod, on="product_id", how="left")

        X_pred = pipeline_pp.transform(df_pred_input)
        proba = model.predict_proba(X_pred)[:, 1]

        df_pred_input["probabilidad"] = proba

        out = f"/opt/airflow/data/predictions/pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_pred_input.to_csv(out, index=False)

        print(f"💾 Predicciones guardadas en: {out}")

    predecir_task = PythonOperator(
        task_id="generar_predicciones",
        python_callable=task_predecir,
        provide_context=True,
        trigger_rule="one_success",
    )

    # FIN
    fin_ok = EmptyOperator(task_id="fin_pipeline")

    (
        ingesta_task >>
        preprocesar_task >>
        detectar_drift_task >>
        [reentrenar_task, usar_modelo_existente_task] >>
        predecir_task >>
        fin_ok
    )