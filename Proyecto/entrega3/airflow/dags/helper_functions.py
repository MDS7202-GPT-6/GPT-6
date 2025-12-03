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


MLFLOW_TRACKING_URI = "http://mlflow:5001"
MLFLOW_EXPERIMENT = "modelo_tiendas_ancla"

DATA_PATH = "/opt/airflow/data"
MODEL_PATH = os.path.join(DATA_PATH, "models")
PRED_PATH = os.path.join(DATA_PATH, "predictions")


def _init_mlflow():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        return True
    except:
        print("⚠️ MLflow no disponible")
        return False


# =============================================================================
# TRANSFORMADORES
# =============================================================================

class GeoClustering(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None

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
        else:
            X["geo_cluster"] = -1

        X["geo_cluster"] = X["geo_cluster"].fillna(-1).astype(int)
        return X


class IQR(BaseEstimator, TransformerMixin):
    def __init__(self, l=1.5):
        self.l = l
        self.q1 = None
        self.q3 = None

    def fit(self, X, y=None):
        self.q1 = X.quantile(0.25)
        self.q3 = X.quantile(0.75)
        self.iqr = self.q3 - self.q1
        return self

    def transform(self, X):
        X = X.copy()
        lower = self.q1 - self.l * self.iqr
        upper = self.q3 + self.l * self.iqr
        for col in X.columns:
            X.loc[(X[col] < lower[col]) | (X[col] > upper[col]), col] = np.nan
        return X


class FeatureAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, df_transacciones=None):
        if df_transacciones is None:
            raise ValueError("df_transacciones requerido")
        X = X.copy()

        df_transacciones = df_transacciones.copy()
        df_transacciones["purchase_date"] = pd.to_datetime(df_transacciones["purchase_date"])

        X["fecha_actual"] = pd.to_datetime(
            X["Año"].astype(str) + "-W" + X["Semana"].astype(str) + "-1",
            format="%G-W%V-%u"
        )

        trans = df_transacciones[["customer_id", "product_id", "purchase_date"]]
        trans = trans.rename(columns={"purchase_date": "fecha_compra"})

        merged = X.merge(trans, on=["customer_id", "product_id"], how="left")
        merged = merged[merged["fecha_compra"] <= merged["fecha_actual"]]

        freq = (
            merged.groupby(["customer_id", "product_id", "Año", "Semana"])
            .size()
            .reset_index(name="frecuencia")
        )
        X = X.merge(freq, on=["customer_id", "product_id", "Año", "Semana"], how="left")
        X["frecuencia"] = X["frecuencia"].fillna(0).astype(int)

        X["trimestre"] = X["fecha_actual"].dt.quarter
        X = X.drop(columns=["fecha_actual"])
        return X


# =============================================================================
# CARGA
# =============================================================================

def load_data():
    df_trans = pd.read_parquet(f"{DATA_PATH}/raw/transacciones.parquet")
    df_cli = pd.read_parquet(f"{DATA_PATH}/raw/clientes.parquet")
    df_prod = pd.read_parquet(f"{DATA_PATH}/raw/productos.parquet")
    return df_trans, df_cli, df_prod


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def preprocess_data(df_transacciones, df_clientes, df_productos):
    print("⚙️ Preprocesando...")

    df_clientes["customer_type"] = df_clientes["customer_type"].astype("string")
    df_productos["brand"] = df_productos["brand"].astype("string")
    df_productos["category"] = df_productos["category"].astype("string")
    df_productos["segment"] = df_productos["segment"].astype("string")
    df_transacciones["purchase_date"] = pd.to_datetime(df_transacciones["purchase_date"])

    df_transacciones = df_transacciones.groupby(
        ["order_id", "product_id", "customer_id", "purchase_date"],
        as_index=False
    )["items"].sum()

    df_transacciones = df_transacciones[df_transacciones["items"] >= 0]
    df_transacciones["Semana"] = df_transacciones["purchase_date"].dt.isocalendar().week
    df_transacciones["Año"] = df_transacciones["purchase_date"].dt.year
    df_transacciones["target"] = 1

    clientes = df_transacciones["customer_id"].unique()
    productos = df_transacciones["product_id"].unique()
    semanas = df_transacciones[["Año", "Semana"]].drop_duplicates()

    full = (
        pd.DataFrame({"customer_id": clientes})
        .merge(pd.DataFrame({"product_id": productos}), how="cross")
        .merge(semanas, how="cross")
    )

    df = full.merge(
        df_transacciones[["customer_id","product_id","Año","Semana","target"]],
        on=["customer_id","product_id","Año","Semana"],
        how="left"
    )
    df["target"] = df["target"].fillna(0).astype(int)

    df = df.merge(df_clientes, on="customer_id")
    df = df.merge(df_productos, on="product_id")
    df = df.drop_duplicates()

    df = df.sort_values("Semana")
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Undersample
    def undersample(X, y):
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        ratio = min(10, len(idx0) // len(idx1))
        size0 = len(idx1) * ratio
        idx0_s = np.random.choice(idx0, size=size0, replace=False)

        idx_final = np.concatenate([idx0_s, idx1])
        np.random.shuffle(idx_final)

        return X.iloc[idx_final], y.iloc[idx_final]

    X_bal, y_bal = undersample(X_train, y_train)

    num_features = ["num_deliver_per_week","num_visit_per_week","size","frecuencia"]
    cat_features = ["customer_type","brand","category","segment","geo_cluster","trimestre"]

    numeric_pipeline = Pipeline([
        ("iqr", IQR()),
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("scaler", MinMaxScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features)
    ])

    def add_feats(X):
        return FeatureAggregator().transform(X, df_transacciones=df_transacciones)

    pipeline_pp = Pipeline([
        ("feats", FunctionTransformer(add_feats)),
        ("geo", GeoClustering(4)),
        ("prep", preprocessor)
    ])

    X_bal_t = pipeline_pp.fit_transform(X_bal)
    X_val_t = pipeline_pp.transform(X_val)
    X_test_t = pipeline_pp.transform(X_test)

    return X_bal_t, X_val_t, y_bal, y_val, X_test_t, y_test, pipeline_pp


# =============================================================================
# DRIFT
# =============================================================================

def detect_drift(df_old, df_new, threshold=0.1):
    metrics = []

    for col in df_old.columns:
        if col not in df_new.columns:
            continue

        if df_old[col].dtype.kind in "iuf" and df_old[col].nunique() >= 10:
            p = ks_2samp(df_old[col].dropna(), df_new[col].dropna()).pvalue
            metrics.append(p)
        else:
            old = df_old[col].value_counts(normalize=True)
            new = df_new[col].value_counts(normalize=True)
            cats = list(set(old.index) | set(new.index))
            p = np.array([old.get(c,0) for c in cats])
            q = np.array([new.get(c,0) for c in cats])
            js = jensenshannon(p,q)
            metrics.append(1-js)

    score = float(np.mean(metrics))
    return score < threshold


# =============================================================================
# OPTUNA
# =============================================================================

def optimize_with_optuna(X_train, X_val, y_train, y_val, n_trials=30):
    import optuna
    from optuna.samplers import TPESampler

    def obj(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42
        }

        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return f1_score(y_val, pred, average="macro")

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials)
    best = study.best_params
    best["random_state"] = 42
    return best


# =============================================================================
# ENTRENAMIENTO FINAL
# =============================================================================

def train_and_log_model(X_train, X_val, y_train, y_val, X_test, y_test, optimize):
    mlflow_ok = _init_mlflow()

    if optimize:
        best_params = optimize_with_optuna(X_train, X_val, y_train, y_val)
    else:
        best_params = {
            "max_depth": 14,
            "min_samples_split": 10,
            "min_samples_leaf": 7,
            "random_state": 42
        }

    print(f"🏋️ Entrenando modelo validación con {best_params}")
    model = DecisionTreeClassifier(**best_params)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    f1 = f1_score(y_val, pred, average="macro")
    acc = accuracy_score(y_val, pred)
    print(f"Validación → F1={f1:.4f}, ACC={acc:.4f}")

    # ================================
    # REENTRENAMIENTO FINAL CON TODOS
    # ================================
    print("🔁 Reentrenando modelo FINAL con TRAIN+VAL+TEST...")

    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate(
        [np.ravel(y_train), np.ravel(y_val), np.ravel(y_test)],
        axis=0
    )

    model_final = DecisionTreeClassifier(**best_params)
    model_final.fit(X_all, y_all)

    print("🏁 Modelo FINAL entrenado correctamente.")
    return model_final
