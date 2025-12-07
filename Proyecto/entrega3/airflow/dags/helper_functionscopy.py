import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import shap
import joblib
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

# ====================================================
# CONFIGURACIONES GLOBALES
# ====================================================

DATA_PATH = "/opt/airflow/data"
MODEL_PATH = os.path.join(DATA_PATH, "models")
PRED_PATH = os.path.join(DATA_PATH, "predictions")
MLFLOW_TRACKING_URI = "http://mlflow:5001"
MLFLOW_EXPERIMENT = "modelo_tiendas_ancla"

# No ejecutar código de MLflow aquí (nivel superior)
# Se inicializará dentro de cada función que lo necesite


def _init_mlflow():
    """Inicializa MLflow solo cuando se necesita (dentro de funciones)"""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        print("MLflow inicializado correctamente")
        return True
    except Exception as e:
        print(f"WARNING: MLflow no disponible: {e}")
        print("WARNING: Continuando sin tracking de MLflow...")
        return False


# ====================================================
# CLASES TRANSFORMADORAS PERSONALIZADAS
# ====================================================

class GeoClustering(BaseEstimator, TransformerMixin):
    """Clustering geográfico basado en coordenadas X, Y"""
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.random_state = random_state

    def fit(self, X, y=None):
        if "X" in X.columns and "Y" in X.columns:
            coords = X[["X", "Y"]].dropna()
            if len(coords) > 0:
                self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
                self.kmeans.fit(coords)
        return self

    def transform(self, X):
        X = X.copy()
        if self.kmeans is not None and "X" in X.columns and "Y" in X.columns:
            mask = X[["X", "Y"]].notna().all(axis=1)
            X.loc[mask, "geo_cluster"] = self.kmeans.predict(X.loc[mask, ["X", "Y"]])
            X["geo_cluster"] = X["geo_cluster"].fillna(-1).astype(int)
        else:
            X["geo_cluster"] = -1
        return X


class IQR(BaseEstimator, TransformerMixin):
    """Eliminación de outliers usando método IQR - reemplaza con NaN, no elimina filas"""
    def __init__(self, l=1.5):
        self.l = l
        self.inferior = None
        self.superior = None

    def fit(self, X, y=None):
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.inferior = q1 - self.l * iqr
        self.superior = q3 + self.l * iqr
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            # Reemplazar outliers con NaN en lugar de eliminar filas
            mask = (X[col] < self.inferior[col]) | (X[col] > self.superior[col])
            X.loc[mask, col] = np.nan
        return X


class FeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Genera features de frecuencia, trimestre y otras agregaciones temporales.
    Requiere df_transacciones como parámetro en transform().
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, df_transacciones=None):
        if df_transacciones is None:
            raise ValueError("Necesitas pasar df_transacciones a transform()")

        X = X.copy()
        df_transacciones = df_transacciones.copy()
        df_transacciones["purchase_date"] = pd.to_datetime(df_transacciones["purchase_date"])

        # Crear fecha_actual para cada fila
        X["fecha_actual"] = pd.to_datetime(
            X["Año"].astype(str) + "-W" + X["Semana"].astype(str) + "-1",
            format="%G-W%V-%u"
        )
        
        # Preparar transacciones solo con columnas necesarias para evitar duplicados
        trans = df_transacciones[["customer_id", "product_id", "purchase_date"]].copy()
        trans = trans.rename(columns={"purchase_date": "fecha_compra"})

        # Merge para obtener compras previas
        merged = X.merge(trans, on=["customer_id", "product_id"], how="left")
        merged = merged[merged["fecha_compra"] <= merged["fecha_actual"]]

        # Frecuencia por producto
        freq = (
            merged.groupby(["customer_id", "product_id", "Año", "Semana"])
            .size()
            .reset_index(name="frecuencia")
        )
        X = X.merge(freq, on=["customer_id", "product_id", "Año", "Semana"], how="left")
        X["frecuencia"] = X["frecuencia"].fillna(0).astype(int)

        # Trimestre
        X["trimestre"] = X["fecha_actual"].dt.quarter

        # Frecuencia por categoría
        if "category" in X.columns:
            freq_cat = (
                merged.groupby(["customer_id", "category", "Año", "Semana"])
                .size()
                .reset_index(name="frecuencia_categoria")
            )
            X = X.merge(freq_cat, on=["customer_id", "category", "Año", "Semana"], how="left")
            X["frecuencia_categoria"] = X["frecuencia_categoria"].fillna(0).astype(int)

        # Frecuencia por marca
        if "brand" in X.columns:
            freq_brand = (
                merged.groupby(["customer_id", "brand", "Año", "Semana"])
                .size()
                .reset_index(name="frecuencia_brand")
            )
            X = X.merge(freq_brand, on=["customer_id", "brand", "Año", "Semana"], how="left")
            X["frecuencia_brand"] = X["frecuencia_brand"].fillna(0).astype(int)

        X = X.drop(columns=["fecha_actual"])
        return X


# ====================================================
# 1. CARGA DE DATOS
# ====================================================

def load_data():
    """
    Carga transacciones, clientes y productos desde /data/raw/
    """
    # Crear directorios si no existen
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(PRED_PATH, exist_ok=True)
    
    trans_path = os.path.join(DATA_PATH, "raw", "transacciones.parquet")
    clientes_path = os.path.join(DATA_PATH, "raw", "clientes.parquet")
    productos_path = os.path.join(DATA_PATH, "raw", "productos.parquet")
    
    print(f"Cargando datos desde {DATA_PATH}/raw ...")
    df_transacciones = pd.read_parquet(trans_path)
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)
    
    print(f"Transacciones: {df_transacciones.shape}")
    print(f"Clientes: {df_clientes.shape}")
    print(f"Productos: {df_productos.shape}")
    
    return df_transacciones, df_clientes, df_productos


# ====================================================
# 2. LIMPIEZA Y TRANSFORMACIÓN
# ====================================================

def preprocess_data(df_transacciones: pd.DataFrame, df_clientes: pd.DataFrame, df_productos: pd.DataFrame):
    """
    Pipeline completo de preprocesamiento según Entrega 1.
    """
    print("Iniciando preprocesamiento...")
    
    # 1. Convertir tipos de datos
    df_clientes["customer_type"] = df_clientes["customer_type"].astype("string")
    df_productos["brand"] = df_productos["brand"].astype("string")
    df_productos["category"] = df_productos["category"].astype("string")
    df_productos["sub_category"] = df_productos["sub_category"].astype("string")
    df_productos["segment"] = df_productos["segment"].astype("string")
    df_productos["package"] = df_productos["package"].astype("string")
    
    df_transacciones["purchase_date"] = pd.to_datetime(df_transacciones["purchase_date"])
    
    # 2. Agrupar y sumar items
    df_transacciones = df_transacciones.groupby(
        ["order_id", "product_id", "customer_id", "purchase_date"],
        as_index=False
    )["items"].sum()
    
    # 3. Filtrar items >= 0
    df_transacciones = df_transacciones[df_transacciones["items"] >= 0]
    
    # 4. Crear variables temporales
    df_transacciones["Semana"] = df_transacciones["purchase_date"].dt.isocalendar().week
    df_transacciones["Año"] = df_transacciones["purchase_date"].dt.year
    
    # 5. Crear target
    df_transacciones["target"] = 1
    
    # 6. Generar todas las combinaciones
    clientes = df_transacciones["customer_id"].unique()
    productos = df_transacciones["product_id"].unique()
    semanas = df_transacciones[["Año", "Semana"]].drop_duplicates()
    
    todasCombinaciones = (
        pd.DataFrame({"customer_id": clientes})
        .merge(pd.DataFrame({"product_id": productos}), how="cross")
        .merge(semanas, how="cross")
    )
    
    # 7. Merge para completar con target=0
    df = todasCombinaciones.merge(
        df_transacciones[["customer_id", "product_id", "Año", "Semana", "target"]],
        on=["customer_id", "product_id", "Año", "Semana"],
        how="left"
    )
    df["target"] = df["target"].fillna(0).astype(int)
    
    # 8. Merge con clientes y productos
    df = df.merge(df_clientes, on="customer_id").merge(df_productos, on="product_id")
    
    # 9. Eliminar duplicados
    df = df.drop_duplicates()
    
    # 10. Imputar valores nulos en X con SimpleImputer
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="most_frequent")
    df[["X"]] = imputer.fit_transform(df[["X"]])
    
    print(f"Datos combinados: {df.shape}")
    
    # 11. Dividir en train/val/test respetando temporalidad
    df_sorted = df.sort_values("Semana")
    n = len(df_sorted)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    # 12. Undersampling para balancear
    def underSamp(Xtrain, ytrain):
        idx_class0 = np.where(ytrain == 0)[0]
        idx_class1 = np.where(ytrain == 1)[0]
        
        # Calcular el tamaño máximo de muestra posible
        max_ratio = len(idx_class0) // len(idx_class1) if len(idx_class1) > 0 else 1
        target_ratio = min(10, max_ratio)  # Usar ratio 10 o el máximo posible
        
        if target_ratio < 1:
            # Si no hay suficientes muestras clase 0, usar todas
            print(f"WARNING: Pocas muestras clase 0 ({len(idx_class0)}), usando todas")
            idx_class0_sampled = idx_class0
        else:
            sample_size = min(len(idx_class1) * target_ratio, len(idx_class0))
            idx_class0_sampled = np.random.choice(idx_class0, size=sample_size, replace=False)
        
        idx_final = np.concatenate([idx_class0_sampled, idx_class1])
        np.random.shuffle(idx_final)
        
        print(f"Balance final: clase 0={len(idx_class0_sampled)}, clase 1={len(idx_class1)}, ratio={len(idx_class0_sampled)/len(idx_class1):.2f}")
        return Xtrain.iloc[idx_final], ytrain.iloc[idx_final]
    
    X_bal, y_bal = underSamp(X_train, y_train)
    
    print(f"Train balanceado: {X_bal.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 13. Definir pipeline de preprocesamiento
    numeric_features = ["num_deliver_per_week", "num_visit_per_week", "size", "frecuencia", "frecuencia_categoria", "frecuencia_brand"]
    categorical_features = ["customer_type", "brand", "category", "segment", "package", "geo_cluster", "trimestre"]
    
    # Usar los mejores hiperparámetros de Optuna
    best_params = {
        'n_clusters': 4,
        'num_scaler': 'minmax',
        'cat_encoder': 'onehot',
        'max_depth': 14,
        'min_samples_split': 10,
        'min_samples_leaf': 7
    }
    
    # Configurar scalers y encoders
    num_scaler = MinMaxScaler()

    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    numeric_pipeline = Pipeline([
        ("iqr", IQR()),
        ("imputer", SimpleImputer(fill_value=0, strategy="most_frequent")),
        ("scaler", num_scaler)
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", cat_encoder)
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    
    def agregar_features(X):
        return FeatureAggregator().transform(X, df_transacciones=df_transacciones)
    
    pipeline_pp = Pipeline([
        ("features", FunctionTransformer(agregar_features)),
        ("geo_cluster", GeoClustering(n_clusters=best_params["n_clusters"])),
        ("preprocessing", preprocessor)
    ])
    
    # 14. Transformar datos
    X_bal_transformed = pipeline_pp.fit_transform(X_bal)
    X_val_transformed = pipeline_pp.transform(X_val)
    
    print(f"Transformación completa: train={X_bal_transformed.shape}, val={X_val_transformed.shape}")
    
    return X_bal_transformed, X_val_transformed, y_bal, y_val, X_test, y_test, pipeline_pp


# ====================================================
# 3. DETECCIÓN DE DRIFT
# ====================================================

def detect_drift(df_old: pd.DataFrame, df_new: pd.DataFrame, threshold: float = 0.1) -> bool:
    """
    Detecta drift entre datasets antiguos y nuevos usando:
      - KS test para numéricas
      - Jensen-Shannon Divergence para categóricas
    """
    metrics = []

    for col in df_old.columns:
        # Validar que la columna existe en ambos sets
        if col not in df_new.columns:
            continue

        # NUMÉRICAS -> KS Test
        if df_old[col].nunique() >= 10 and df_new[col].nunique() >= 10 and np.issubdtype(df_old[col].dtype, np.number):
            try:
                p = ks_2samp(df_old[col].dropna(), df_new[col].dropna()).pvalue
                metrics.append(p)  
            except Exception:
                continue

        # CATEGÓRICAS -> Jensen Shannon
        else:
            old_counts = df_old[col].value_counts(normalize=True)
            new_counts = df_new[col].value_counts(normalize=True)

            # Alinear categorías entre ambos dataframes
            all_cats = sorted(set(old_counts.index) | set(new_counts.index))
            p = np.array([old_counts.get(cat, 0) for cat in all_cats])
            q = np.array([new_counts.get(cat, 0) for cat in all_cats])

            js = jensenshannon(p, q)

            # Convertimos JS a "similaridad" como si fuera p-value (1-js)
            metrics.append(1 - js)

    # Promedio de las métricas
    avg_score = np.mean(metrics) if len(metrics) > 0 else 1.0

    # Si el score promedio cae por debajo del umbral -> hay drift
    drift = avg_score < threshold

    print(f"Score promedio={avg_score:.4f} -> Drift={drift}")
    return drift


# ====================================================
# 4. OPTIMIZACIÓN CON OPTUNA
# ====================================================

def optimize_with_optuna(X_train, X_val, y_train, y_val, n_trials=30):
    """
    Optimiza hiperparámetros usando Optuna con conjunto de validación.
    Se ejecuta solo cuando NO existe modelo previo.
    """
    import optuna
    from optuna.samplers import TPESampler
    
    print(f"Iniciando optimización con Optuna ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
        
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        return f1
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    
    print(f"Optimización completada!")
    print(f"Mejores hiperparámetros: {best_params}")
    print(f"Mejor F1-score: {study.best_value:.4f}")
    
    return best_params


# ====================================================
# 5. ENTRENAMIENTO CON DecisionTree + MLflow + SHAP
# ====================================================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import numpy as np
import pandas as pd
import joblib
import mlflow

def train_and_log_model(X_train, X_val, y_train, y_val, optimize=False):
    """
    Entrena DecisionTreeClassifier y muestra métricas detalladas en consola.
    """

    # Inicializar MLflow si está disponible
    mlflow_available = _init_mlflow()

    # Hiperparámetros
    if optimize:
        print("\n🎯 Optimizando hiperparámetros con Optuna…")
        best_params = optimize_with_optuna(X_train, X_val, y_train, y_val)
    else:
        print("\n♻️ Reentrenamiento → Usando hiperparámetros predefinidos")
        best_params = {
            'max_depth': 14,
            'min_samples_split': 10,
            'min_samples_leaf': 8,
            'random_state': 42
        }

    print("\n==============================")
    print("🧠 ENTRENANDO MODELO")
    print("==============================")
    print(best_params)

    model = DecisionTreeClassifier(**best_params)
    model.fit(X_train, y_train)

    # ============================
    # 🔍 PREDICCIONES VALIDACIÓN
    # ============================
    print("\n==============================")
    print("📊 MÉTRICAS VALIDACIÓN (VAL)")
    print("==============================")

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print("\n--- Reporte de clasificación ---")
    print(classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    print("\n--- Matriz de confusión (VAL) ---")
    print(pd.DataFrame(cm,
                       index=["Real 0", "Real 1"],
                       columns=["Pred 0", "Pred 1"]))

    print("\n--- Distribución de predicciones (VAL) ---")
    unique, counts = np.unique(y_pred, return_counts=True)
    print(dict(zip(unique, counts)))

    print("\n--- Tasa real vs predicha de positivos (VAL) ---")
    print(f"Positivos reales:   {y_val.mean():.4f}")
    print(f"Positivos predichos:{y_pred.mean():.4f}")

    # ============================
    # 🔍 PREDICCIONES TRAIN (opcional)
    # ============================
    print("\n==============================")
    print("📊 MÉTRICAS TRAIN BALANCEADO")
    print("==============================")

    train_pred = model.predict(X_train)

    print(f"Accuracy Train:  {accuracy_score(y_train, train_pred):.4f}")
    print(f"Recall Train:    {recall_score(y_train, train_pred):.4f}")
    print(f"F1 Train:        {f1_score(y_train, train_pred):.4f}")
    print("\n--- Matriz de confusión (TRAIN) ---")
    print(confusion_matrix(y_train, train_pred))

    # ============================
    # 🔍 MLflow
    # ============================
    if mlflow_available:
        try:
            with mlflow.start_run(run_name=f"DecisionTree_{datetime.now().strftime('%Y%m%d_%H%M')}"):

                mlflow.log_params(best_params)
                mlflow.log_metric("val_accuracy", acc)
                mlflow.log_metric("val_precision", prec)
                mlflow.log_metric("val_recall", rec)
                mlflow.log_metric("val_f1", f1)

                # Clasificación por clase
                rep = classification_report(y_val, y_pred, output_dict=True)
                for label in ["0", "1"]:
                    mlflow.log_metric(f"precision_class_{label}", rep[label]["precision"])
                    mlflow.log_metric(f"recall_class_{label}", rep[label]["recall"])
                    mlflow.log_metric(f"f1_class_{label}", rep[label]["f1-score"])

                # Guardar modelo en MLflow
                mlflow.sklearn.log_model(model, "model")

                print("\n📦 Modelo logueado en MLflow.")

        except Exception as e:
            print(f"⚠️ Error al loguear MLflow: {e}")

    # ============================
    # 💾 GUARDAR MODELO LOCAL
    # ============================
    model_path = os.path.join(MODEL_PATH, f"modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    joblib.dump(model, model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")

    print("\n==============================")
    print(f"🏁 Entrenamiento completado | F1_val={f1:.4f} | ACC_val={acc:.4f}")
    print("==============================\n")

    return model


# ====================================================
# 📈 5. GENERACIÓN DE PREDICCIONES
# ====================================================

def predict_future_week(model, df_features: pd.DataFrame):
    """
    Usa el modelo para predecir la próxima semana (t+2).
    Supone que df_features contiene la última semana t+1.
    """
    preds = model.predict_proba(df_features)[:, 1]
    df_pred = df_features.copy()
    df_pred["prediccion_compra"] = preds
    df_pred["semana_predicha"] = df_pred["semana"].max() + 1

    out_path = os.path.join(PRED_PATH, f"predicciones_{datetime.now().strftime('%Y%m%d')}.parquet")
    df_pred.to_parquet(out_path, index=False)
    print(f"Predicciones guardadas en {out_path}")

    return df_pred