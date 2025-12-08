from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import os
import joblib
from helper_functionscopy import (
    load_data,
    preprocess_data,
    detect_drift,
    train_and_log_model,
    predict_future_week
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
    dag_id="pipeline_modelo_previo",
    default_args=default_args,
    description="Pipeline de modelado con DecisionTree + Optuna + MLflow + SHAP (Entrega 1)",
    schedule_interval="@weekly",   # cada semana
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["mlops", "decisiontree", "optuna", "mlflow"],
) as dag:



    def task_ingesta(**context):
        df_transacciones, df_clientes, df_productos = load_data()
        context["ti"].xcom_push(key="df_transacciones", value=df_transacciones.to_json())
        context["ti"].xcom_push(key="df_clientes", value=df_clientes.to_json())
        context["ti"].xcom_push(key="df_productos", value=df_productos.to_json())

    ingesta_task = PythonOperator(
        task_id="ingestar_datos",
        python_callable=task_ingesta,
        provide_context=True,
    )



    def task_preprocesar(**context):
        import pandas as pd
        df_transacciones = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_transacciones"))
        df_clientes = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_clientes"))
        df_productos = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_productos"))
        
        X_train, X_val, y_train, y_val, X_test, y_test, pipeline_pp = preprocess_data(
            df_transacciones, df_clientes, df_productos
        )
        
        # Guardar como numpy arrays (más eficiente que JSON para matrices grandes)
        import numpy as np
        np.save('/tmp/X_train.npy', X_train)
        np.save('/tmp/X_val.npy', X_val)
        np.save('/tmp/y_train.npy', y_train)
        np.save('/tmp/y_val.npy', y_val)
        
        # Guardar pipeline
        import joblib
        joblib.dump(pipeline_pp, '/tmp/pipeline_pp.pkl')
        # TAMBIÉN guardar en el directorio de modelos para la app
        joblib.dump(pipeline_pp, '/opt/airflow/data/models/pipeline_pp.pkl')
        print("✅ Pipeline guardado en /opt/airflow/data/models/pipeline_pp.pkl")

    preprocesar_task = PythonOperator(
        task_id="preprocesar_datos",
        python_callable=task_preprocesar,
        provide_context=True,
    )


    def task_detectar_drift(**context):
        import pandas as pd
        import os

        # Carpeta donde guardas el snapshot OLD
        old_dir = "/opt/airflow/data/raw_old"
        new_dir = "/opt/airflow/data/raw"   # la carpeta normal raw

        os.makedirs(old_dir, exist_ok=True)

        # Cargar new
        df_new = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos",
                                                    key="df_transacciones"))

        # Si no existe snapshot → primera corrida
        old_path = os.path.join(old_dir, "df_transacciones_old.parquet")
        if not os.path.exists(old_path):
            print("⚠️ No existe raw_old → primera ejecución → reentrenar")
            return "reentrenar_modelo"

        # Cargar old
        df_old = pd.read_parquet(old_path)

        print("Comparando raw_old vs raw...")

        drift = detect_drift(
            df_old[["customer_id","product_id","purchase_date","items"]],
            df_new[["customer_id","product_id","purchase_date","items"]],
            threshold=0.1
        )

        if drift:
            print("🔴 Drift detectado → reentrenar")
            return "reentrenar_modelo"
        else:
            print("🟢 No drift → usar modelo existente")
            return "usar_modelo_existente"

    detectar_drift_task = BranchPythonOperator(
        task_id="detectar_drift",
        python_callable=task_detectar_drift,
        provide_context=True,
    )




    def task_reentrenar(**context):
        import numpy as np
        import os
        import pandas as pd
        
        # Cargar datos transformados
        X_train = np.load('/tmp/X_train.npy', allow_pickle=True)
        X_val = np.load('/tmp/X_val.npy', allow_pickle=True)
        y_train = np.load('/tmp/y_train.npy', allow_pickle=True)
        y_val = np.load('/tmp/y_val.npy', allow_pickle=True)
        
        # Verificar si existe modelo previo para decidir si optimizar
        model_dir = "/opt/airflow/data/models"
        modelos = [f for f in os.listdir(model_dir) if f.endswith(".pkl") and not f.startswith("pipeline")]
        
        # Si NO existe modelo previo → optimizar con Optuna
        # Si existe modelo previo → usar hiperparámetros de Entrega 1
        optimize = len(modelos) == 0
        
        if optimize:
            print("🎯 Primera ejecución detectada: Se optimizará con Optuna")
        else:
            print("♻️ Reentrenamiento por drift: Se usarán hiperparámetros pre-optimizados")
        
        model = train_and_log_model(X_train, X_val, y_train, y_val, optimize=optimize)
        
        model_path = f"/opt/airflow/data/models/modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(model, model_path)
        context["ti"].xcom_push(key="model_path", value=model_path)
        
        print(f"✅ Modelo guardado en: {model_path}")

        df_new = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos",
                                               key="df_transacciones"))

        snapshot_path = "/opt/airflow/data/raw_old/df_transacciones_old.parquet"
        df_new.to_parquet(snapshot_path)
        print(f"📦 Snapshot OLD actualizado en: {snapshot_path}")

    reentrenar_task = PythonOperator(
        task_id="reentrenar_modelo",
        python_callable=task_reentrenar,
        provide_context=True,
    )



    def task_usar_modelo_existente(**context):
        """
        Si no hay drift, usa el modelo más reciente disponible en /models.
        """
        model_dir = "/opt/airflow/data/models"
        modelos = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pkl") and not f.startswith("pipeline")]
        if not modelos:
            raise FileNotFoundError("⚠️ No se encontraron modelos previos entrenados.")
        latest_model = max(modelos, key=os.path.getctime)
        context["ti"].xcom_push(key="model_path", value=latest_model)
        print(f"✅ Usando modelo existente: {latest_model}")

    usar_modelo_existente_task = PythonOperator(
        task_id="usar_modelo_existente",
        python_callable=task_usar_modelo_existente,
        provide_context=True,
    )




    def task_predecir(**context):
        import pandas as pd
        import joblib
        import numpy as np
        from itertools import product
        import datetime

        print("\n==================== PREDICCIÓN ====================\n")

        # ----------------------------------------------------
        # 1️⃣ CARGAR MODELO
        # ----------------------------------------------------
        model_path = context["ti"].xcom_pull(
            task_ids=["reentrenar_modelo", "usar_modelo_existente"],
            key="model_path"
        )
        model_path = [m for m in model_path if m is not None][0]
        print(f"📦 Cargando modelo desde: {model_path}")
        model = joblib.load(model_path)

        # ----------------------------------------------------
        # 2️⃣ CARGAR PIPELINE DE FEATURES
        # ----------------------------------------------------
        pipeline_pp = joblib.load('/tmp/pipeline_pp.pkl')
        print("📦 Pipeline de preprocesamiento cargado.")

        # ----------------------------------------------------
        # 3️⃣ CARGAR DATOS BASE (transacciones, clientes, productos)
        # ----------------------------------------------------
        df_trans = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_transacciones"))
        df_clientes = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_clientes"))
        df_productos = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_productos"))

        # Convertir fecha desde UNIX-ms (como vimos en tus datos)
        df_trans["purchase_date"] = pd.to_datetime(df_trans["purchase_date"], unit="ms", errors="coerce")

        if df_trans["purchase_date"].isna().all():
            raise ValueError("❌ Todas las fechas en transacciones son inválidas.")

        print("Min fecha:", df_trans["purchase_date"].min())
        print("Max fecha:", df_trans["purchase_date"].max())

        # ----------------------------------------------------
        # 4️⃣ AÑO Y SEMANA SEGÚN TU LÓGICA (Año = .dt.year)
        # ----------------------------------------------------
        df_trans["Año_cal"] = df_trans["purchase_date"].dt.year
        iso = df_trans["purchase_date"].dt.isocalendar()
        df_trans["Semana"] = iso.week.astype(int)

        current_year = int(df_trans["Año_cal"].max())
        current_week = int(df_trans.loc[df_trans["Año_cal"] == current_year, "Semana"].max())

        print(f"Última fecha real = {df_trans['purchase_date'].max()} → Año_cal={current_year}, Semana={current_week}")

        # Próxima semana usando AÑO CALENDARIO
        next_year = current_year
        next_week = current_week + 1

        # Si te pasas de 53, reseteas a 1 y avanzas de año calendario
        if next_week > 53:
            next_week = 1
            next_year = current_year + 1

        print(f"🔮 Prediciendo SOLO para Semana={next_week}, Año={next_year} (Año calendario)")

        # ----------------------------------------------------
        # 5️⃣ CREAR UNIVERSO cliente × producto
        #    (SOLO los que existen realmente en transacciones)
        # ----------------------------------------------------
        clientes = df_trans["customer_id"].dropna().unique()
        productos = df_trans["product_id"].dropna().unique()

        df_pred_input = pd.DataFrame(product(clientes, productos), columns=["customer_id", "product_id"])
        df_pred_input["Semana"] = next_week
        df_pred_input["Año"] = next_year

        print(f"Combinaciones generadas: {len(df_pred_input):,}")

        # ----------------------------------------------------
        # 6️⃣ AGREGAR METADATA NECESARIA PARA EL PIPELINE
        # ----------------------------------------------------
        df_pred_input = df_pred_input.merge(df_clientes, on="customer_id", how="left")
        df_pred_input = df_pred_input.merge(df_productos, on="product_id", how="left")

        # ----------------------------------------------------
        # 7️⃣ TRANSFORMAR FEATURES + PREDECIR
        # ----------------------------------------------------
        X_pred = pipeline_pp.transform(df_pred_input)
        proba = model.predict_proba(X_pred)[:, 1]
        df_pred_input["probabilidad_compra"] = proba

        print("\n📊 Stats probabilidad (predicción futura):")
        print(f"min: {proba.min():.4f} | max: {proba.max():.4f}")
        print("percentiles 5, 50, 95:", np.percentile(proba, [5, 50, 95]))

        # ----------------------------------------------------
        # 8️⃣ APLICAR UMBRAL Y GUARDAR SOLO LOS "SÍ COMPRA"
        # ----------------------------------------------------
        UMBRAL = 0.5  # 🔧 AJUSTA ESTE VALOR según tu análisis de val

        df_positivos = df_pred_input[df_pred_input["probabilidad_compra"] >= UMBRAL]\
            .sort_values("probabilidad_compra", ascending=False)

        print(f"\n🔎 Con umbral = {UMBRAL}, positivos futuros predichos: {len(df_positivos):,} "
            f"({len(df_positivos) / len(df_pred_input):.4%} del total)")

        # CSV FINAL: SOLO customer_id, product_id
        df_final = df_positivos[["customer_id", "product_id"]]

        pred_path = f"/opt/airflow/data/predictions/pred_semana_{next_year}_{next_week}.csv"
        df_final.to_csv(pred_path, index=False)
        print(f"💾 Predicciones guardadas en: {pred_path}")

        # ----------------------------------------------------
        # 9️⃣ CREAR future_PREVIO.csv = compras reales de la semana siguiente
        # ----------------------------------------------------
        df_trans["Año"] = df_trans["Año_cal"]  # para que el nombre coincida con tu lógica
        df_futuro = df_trans[
            (df_trans["Año"] == next_year) &
            (df_trans["Semana"] == next_week)
        ][["customer_id", "product_id"]].drop_duplicates()

        futuro_path = "/opt/airflow/data/predictions/future_PREVIO.csv"
        df_futuro.to_csv(futuro_path, index=False)

        print(f"💾 future_PREVIO.csv creado con {len(df_futuro)} compras reales futuras.")
        print("\n==================== FIN PREDICCIÓN ====================\n")

    predecir_task = PythonOperator(
        task_id="generar_predicciones",
        python_callable=task_predecir,
        provide_context=True,
        trigger_rule="one_success"  # Ejecutar si cualquiera de las 2 tareas anteriores tuvo éxito
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