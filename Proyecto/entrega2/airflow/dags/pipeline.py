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
    predict_future_week
)

# ====================================================
# ⚙️ CONFIGURACIÓN DEL DAG
# ====================================================

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
    description="Pipeline de modelado con DecisionTree + Optuna + MLflow + SHAP (Entrega 1)",
    schedule_interval="@weekly",   # cada semana
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["mlops", "decisiontree", "optuna", "mlflow"],
) as dag:

    # ====================================================
    # 🧱 1️⃣ INGESTA DE DATOS
    # ====================================================

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


    # ====================================================
    # 🧹 2️⃣ PREPROCESAMIENTO
    # ====================================================

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


    # ====================================================
    # ⚖️ 3️⃣ DETECCIÓN DE DRIFT
    # ====================================================

    def task_detectar_drift(**context):
        import pandas as pd
        
        # 1. Verificar si existe un modelo previo
        model_dir = "/opt/airflow/data/models"
        os.makedirs(model_dir, exist_ok=True)
        modelos = [f for f in os.listdir(model_dir) if f.endswith(".pkl") and not f.startswith("pipeline")]
        
        if not modelos:
            print("⚠️ No existe modelo previo → REENTRENAR")
            return "reentrenar_modelo"
        
        # 2. Si existe modelo, detectar drift
        print(f"✅ Modelo previo encontrado ({len(modelos)} modelos) → Detectando drift...")
        df_old = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_transacciones"))
        df_new = df_old.copy()

        drift = detect_drift(df_old[["customer_id", "product_id"]], 
                           df_new[["customer_id", "product_id"]], 
                           threshold=0.1)
        
        if drift:
            print("🔴 DRIFT DETECTADO → REENTRENAR")
            return "reentrenar_modelo"
        else:
            print("🟢 NO HAY DRIFT → USAR MODELO EXISTENTE")
            return "usar_modelo_existente"

    detectar_drift_task = BranchPythonOperator(
        task_id="detectar_drift",
        python_callable=task_detectar_drift,
        provide_context=True,
    )


    # ====================================================
    # 🧠 4️⃣ REENTRENAMIENTO (si hay drift o no existe modelo)
    # ====================================================

    def task_reentrenar(**context):
        import numpy as np
        import os
        
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

    reentrenar_task = PythonOperator(
        task_id="reentrenar_modelo",
        python_callable=task_reentrenar,
        provide_context=True,
    )


    # ====================================================
    # 🧰 5️⃣ USAR MODELO EXISTENTE (si NO hay drift)
    # ====================================================

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


    # ====================================================
    # 📈 6️⃣ GENERACIÓN DE PREDICCIONES
    # ====================================================

    def task_predecir(**context):
        import numpy as np
        import pandas as pd
        import joblib
        
        # Obtener model_path de cualquiera de las dos tareas previas
        model_path = context["ti"].xcom_pull(task_ids=["reentrenar_modelo", "usar_modelo_existente"], key="model_path")
        # Filtrar None y tomar el primero válido
        model_path = [m for m in model_path if m is not None][0]
        
        print(f"📦 Cargando modelo desde: {model_path}")
        model = joblib.load(model_path)
        
        # Cargar el pipeline de preprocesamiento
        print(f"📦 Cargando pipeline de preprocesamiento...")
        pipeline_pp = joblib.load('/tmp/pipeline_pp.pkl')

        # Cargar datos originales para crear features de predicción
        df_transacciones = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_transacciones"))
        df_clientes = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_clientes"))
        df_productos = pd.read_json(context["ti"].xcom_pull(task_ids="ingestar_datos", key="df_productos"))
        
        # Convertir purchase_date a datetime
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])
        
        # Crear Semana y Año si no existen
        if 'Semana' not in df_transacciones.columns:
            df_transacciones['Semana'] = df_transacciones['purchase_date'].dt.isocalendar().week
            df_transacciones['Año'] = df_transacciones['purchase_date'].dt.year
        
        # Obtener última semana del dataset
        max_semana = df_transacciones['Semana'].max()
        max_año = df_transacciones['Año'].max()
        
        # Crear combinaciones para la próxima semana (predicción)
        # Tomamos una muestra de clientes y productos para no explotar la memoria
        sample_customers = df_clientes['customer_id'].sample(min(100, len(df_clientes)), random_state=42)
        sample_products = df_productos['product_id'].sample(min(50, len(df_productos)), random_state=42)
        
        # Generar combinaciones
        from itertools import product
        combinations = list(product(sample_customers, sample_products))
        
        df_pred_input = pd.DataFrame(combinations, columns=['customer_id', 'product_id'])
        df_pred_input['Semana'] = max_semana + 1
        df_pred_input['Año'] = max_año
        
        # Merge con información de clientes y productos
        df_pred_input = df_pred_input.merge(df_clientes, on='customer_id', how='left')
        df_pred_input = df_pred_input.merge(df_productos, on='product_id', how='left')
        
        print(f"🔮 Generando predicciones para {len(df_pred_input)} combinaciones cliente-producto...")
        
        # Aplicar pipeline de preprocesamiento
        X_pred_transformed = pipeline_pp.transform(df_pred_input)
        
        # Generar predicciones (probabilidades)
        y_pred_proba = model.predict_proba(X_pred_transformed)[:, 1]
        
        # Crear DataFrame de resultados
        df_pred_output = df_pred_input[['customer_id', 'product_id', 'Semana', 'Año']].copy()
        df_pred_output['probabilidad_compra'] = y_pred_proba
        
        # Ordenar por probabilidad descendente
        df_pred_output = df_pred_output.sort_values('probabilidad_compra', ascending=False)
        
        print(f"✅ Predicciones generadas: {df_pred_output.shape}")
        print(f"📊 Top 5 predicciones:\n{df_pred_output.head()}")
        
        # Guardar predicciones
        pred_path = f"/opt/airflow/data/predictions/pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_pred_output.to_csv(pred_path, index=False)
        print(f"💾 Predicciones guardadas en: {pred_path}")

    predecir_task = PythonOperator(
        task_id="generar_predicciones",
        python_callable=task_predecir,
        provide_context=True,
        trigger_rule="one_success"  # Ejecutar si cualquiera de las 2 tareas anteriores tuvo éxito
    )


    # ====================================================
    # ✅ 7️⃣ FINALIZAR PIPELINE
    # ====================================================

    fin_ok = EmptyOperator(task_id="fin_pipeline")


    # ====================================================
    # 🔗 DEPENDENCIAS
    # ====================================================

    (
        ingesta_task
        >> preprocesar_task
        >> detectar_drift_task
        >> [reentrenar_task, usar_modelo_existente_task]
        >> predecir_task
        >> fin_ok
    )