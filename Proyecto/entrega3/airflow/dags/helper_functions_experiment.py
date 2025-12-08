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
from sklearn.ensemble import RandomForestClassifier
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
    except Exception:
        print("⚠️ MLflow no disponible")
        return False


class GeoClustering(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None

    def fit(self, X, y=None):
        if "X" in X.columns and "Y" in X.columns:
            coords = X[["X", "Y"]].dropna()
            if len(coords) > 0:
                self.kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init=10,
                )
                self.kmeans.fit(coords)
        return self

    def transform(self, X):
        X = X.copy()
        if self.kmeans is not None and "X" in X.columns and "Y" in X.columns:
            mask = X[["X", "Y"]].notna().all(axis=1)
            X.loc[mask, "geo_cluster"] = self.kmeans.predict(
                X.loc[mask, ["X", "Y"]]
            )
        else:
            X["geo_cluster"] = -1

        X["geo_cluster"] = X["geo_cluster"].fillna(-1).astype(int)
        return X


class IQR(BaseEstimator, TransformerMixin):
    def __init__(self, l=1.5):
        self.l = l
        self.q1 = None
        self.q3 = None
        self.iqr = None

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
    """
    Agrega:
    - frecuencia de compra (cliente-producto-semana)
    - frecuencia_categoria (cliente-categoría-semana)
    - frecuencia_brand (cliente-marca-semana)
    - trimestre
    - tiempo_ultima_compra_dias
    - tiempo_promedio_recompra_dias
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, df_transacciones=None):
        if df_transacciones is None:
            raise ValueError("df_transacciones requerido")

        X = X.copy()
        df_transacciones = df_transacciones.copy()
        df_transacciones["purchase_date"] = pd.to_datetime(
            df_transacciones["purchase_date"]
        )

        # Fecha "actual" asociada a cada fila (lunes de la semana ISO)
        X["fecha_actual"] = pd.to_datetime(
            X["Año"].astype(str) + "-W" + X["Semana"].astype(str) + "-1",
            format="%G-W%V-%u",
        )

        # Tabla base de transacciones
        trans = df_transacciones[["customer_id", "product_id", "purchase_date"]]
        trans = trans.rename(columns={"purchase_date": "fecha_compra"})

        # Expandir por (cliente, producto) y filtrar por fecha_actual
        merged = X.merge(trans, on=["customer_id", "product_id"], how="left")
        merged = merged[merged["fecha_compra"] <= merged["fecha_actual"]]

        # -----------------------------
        # Frecuencia cliente-producto
        # -----------------------------
        freq = (
            merged.groupby(
                ["customer_id", "product_id", "Año", "Semana"],
                as_index=False,
            )["fecha_compra"]
            .size()
            .rename(columns={"size": "frecuencia"})
        )

        X = X.merge(
            freq, on=["customer_id", "product_id", "Año", "Semana"], how="left"
        )
        X["frecuencia"] = X["frecuencia"].fillna(0).astype(int)

        # -----------------------------
        # Frecuencia por categoría
        # -----------------------------
        if "category" in X.columns:
            freq_cat = (
                merged.groupby(
                    ["customer_id", "category", "Año", "Semana"],
                    as_index=False,
                )["fecha_compra"]
                .size()
                .rename(columns={"size": "frecuencia_categoria"})
            )
            X = X.merge(
                freq_cat,
                on=["customer_id", "category", "Año", "Semana"],
                how="left",
            )
            X["frecuencia_categoria"] = (
                X["frecuencia_categoria"].fillna(0).astype(int)
            )

        # -----------------------------
        # Frecuencia por brand
        # -----------------------------
        if "brand" in X.columns:
            freq_brand = (
                merged.groupby(
                    ["customer_id", "brand", "Año", "Semana"],
                    as_index=False,
                )["fecha_compra"]
                .size()
                .rename(columns={"size": "frecuencia_brand"})
            )
            X = X.merge(
                freq_brand,
                on=["customer_id", "brand", "Año", "Semana"],
                how="left",
            )
            X["frecuencia_brand"] = (
                X["frecuencia_brand"].fillna(0).astype(int)
            )

        # -----------------------------
        # Tiempo desde última compra (en días)
        # 9999 si nunca ha comprado ese producto
        # -----------------------------
        if not merged.empty:
            last_purchase = (
                merged.groupby(
                    [
                        "customer_id",
                        "product_id",
                        "Año",
                        "Semana",
                        "fecha_actual",
                    ],
                    as_index=False,
                )["fecha_compra"]
                .max()
                .rename(columns={"fecha_compra": "ultima_compra"})
            )

            X = X.merge(
                last_purchase,
                on=["customer_id", "product_id", "Año", "Semana", "fecha_actual"],
                how="left",
            )

            X["tiempo_ultima_compra_dias"] = (
                X["fecha_actual"] - X["ultima_compra"]
            ).dt.days
            X["tiempo_ultima_compra_dias"] = X[
                "tiempo_ultima_compra_dias"
            ].fillna(9999).astype(int)
        else:
            # Nadie ha comprado nada
            X["tiempo_ultima_compra_dias"] = 9999

        # -----------------------------
        # Tiempo promedio de recompra (cliente-producto)
        # 9999 si nunca o solo una compra
        # -----------------------------
        def _avg_repurchase_days(s):
            s = s.sort_values().drop_duplicates()
            if len(s) < 2:
                return np.nan
            diffs = s.diff().dt.days.iloc[1:]
            return diffs.mean()

        avg_rep = (
            df_transacciones.groupby(["customer_id", "product_id"])[
                "purchase_date"
            ]
            .apply(_avg_repurchase_days)
            .reset_index(name="tiempo_promedio_recompra_dias")
        )

        X = X.merge(
            avg_rep,
            on=["customer_id", "product_id"],
            how="left",
        )
        X["tiempo_promedio_recompra_dias"] = (
            X["tiempo_promedio_recompra_dias"].fillna(9999).astype(int)
        )

        # Trimestre
        X["trimestre"] = X["fecha_actual"].dt.quarter

        # Limpieza
        X = X.drop(columns=["fecha_actual", "ultima_compra"], errors="ignore")

        return X


def load_data():
    df_trans = pd.read_parquet(f"{DATA_PATH}/raw/transacciones.parquet")
    df_cli = pd.read_parquet(f"{DATA_PATH}/raw/clientes.parquet")
    df_prod = pd.read_parquet(f"{DATA_PATH}/raw/productos.parquet")
    return df_trans, df_cli, df_prod


def preprocess_data(df_transacciones, df_clientes, df_productos):
    print("⚙️ Preprocesando...")

    df_clientes["customer_type"] = df_clientes["customer_type"].astype("string")
    df_productos["brand"] = df_productos["brand"].astype("string")
    df_productos["category"] = df_productos["category"].astype("string")
    df_productos["segment"] = df_productos["segment"].astype("string")
    df_transacciones["purchase_date"] = pd.to_datetime(
        df_transacciones["purchase_date"]
    )

    # Agregar items por order_id, product_id, customer_id, fecha
    df_transacciones = df_transacciones.groupby(
        ["order_id", "product_id", "customer_id", "purchase_date"],
        as_index=False,
    )["items"].sum()

    # Filtrar items negativos
    df_transacciones = df_transacciones[df_transacciones["items"] >= 0]

    # Semana y Año
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
        df_transacciones[["customer_id", "product_id", "Año", "Semana", "target"]],
        on=["customer_id", "product_id", "Año", "Semana"],
        how="left",
    )
    df["target"] = df["target"].fillna(0).astype(int)

    # Join con clientes y productos
    df = df.merge(df_clientes, on="customer_id")
    df = df.merge(df_productos, on="product_id")
    df = df.drop_duplicates()

    # Orden temporal
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

    # =====================
    # Undersampling
    # =====================
    def undersample(X, y):
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]

        if len(idx1) == 0:
            return X, y

        ratio = min(10, len(idx0) // len(idx1)) if len(idx0) > 0 else 1
        size0 = len(idx1) * ratio
        if size0 > len(idx0):
            size0 = len(idx0)

        idx0_s = np.random.choice(idx0, size=size0, replace=False)
        idx_final = np.concatenate([idx0_s, idx1])
        np.random.shuffle(idx_final)

        return X.iloc[idx_final], y.iloc[idx_final]

    X_bal, y_bal = undersample(X_train, y_train)

    # =====================
    # Pipelines
    # =====================
    num_features = [
        "num_deliver_per_week",
        "num_visit_per_week",
        "size",
        "frecuencia",
        "tiempo_ultima_compra_dias",
        "tiempo_promedio_recompra_dias",
    ]
    cat_features = [
        "customer_type",
        "brand",
        "category",
        "segment",
        "geo_cluster",
        "trimestre",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("iqr", IQR()),
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ]
    )

    def add_feats(X):
        return FeatureAggregator().transform(X, df_transacciones=df_transacciones)

    pipeline_pp = Pipeline(
        steps=[
            ("feats", FunctionTransformer(add_feats)),
            ("geo", GeoClustering(4)),
            ("prep", preprocessor),
        ]
    )

    X_bal_t = pipeline_pp.fit_transform(X_bal)
    X_val_t = pipeline_pp.transform(X_val)
    X_test_t = pipeline_pp.transform(X_test)

    # También transformamos TODO el dataset para reentrenar final
    X_all = pipeline_pp.transform(df.drop(columns=["target"]).reset_index(drop=True))
    y_all = df["target"].reset_index(drop=True)

    return X_bal_t, X_val_t, y_bal, y_val, X_test_t, y_test, pipeline_pp, X_all, y_all


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
            p = np.array([old.get(c, 0) for c in cats])
            q = np.array([new.get(c, 0) for c in cats])
            js = jensenshannon(p, q)
            metrics.append(1 - js)

    score = float(np.mean(metrics)) if metrics else 1.0
    return score < threshold


def optimize_with_optuna(X_train, X_val, y_train, y_val, n_trials=30):
    import optuna
    from optuna.samplers import TPESampler

    def obj(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42,
        }

        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return f1_score(y_val, pred, average="binary")

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials)
    best = study.best_params
    best["random_state"] = 42
    return best


def train_and_log_model(X_train, X_val, y_train, y_val, X_test, y_test, optimize):
    mlflow_ok = _init_mlflow()

    if optimize:
        best_params = optimize_with_optuna(X_train, X_val, y_train, y_val)
    else:
        best_params = {
            "max_depth": 14,
            "min_samples_split": 10,
            "min_samples_leaf": 7,
            "random_state": 42,
        }

    print(f"🏋️ Entrenando modelo validación con {best_params}")
    model = DecisionTreeClassifier(**best_params)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    f1 = f1_score(y_val, pred, average="binary")
    acc = accuracy_score(y_val, pred)
    print(f"Validación → F1={f1:.4f}, ACC={acc:.4f}")

    print("🔁 Reentrenando modelo FINAL con TRAIN+VAL+TEST...")

    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate(
        [np.ravel(y_train), np.ravel(y_val), np.ravel(y_test)], axis=0
    )

    model_final = DecisionTreeClassifier(**best_params)
    model_final.fit(X_all, y_all)

    print("🏁 Modelo FINAL entrenado correctamente.")
    return model_final


# ================================
# Experiment helpers: threshold and model search
# ================================


def optimize_threshold(y_true, y_proba, thresholds=None):
    import numpy as _np
    from sklearn.metrics import f1_score as _f1

    if thresholds is None:
        thresholds = _np.linspace(0.01, 0.99, 99)

    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = _f1(y_true, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, float(best_f1)


def model_search(X_train, X_val, y_train, y_val, random_state=42, optuna_trials=40):
    """
    Búsqueda híbrida:
    1. Grid Search rápido para warm-start.
    2. Optuna para refinar el mejor modelo.
    3. Calcula threshold óptimo.
    """

    print("\n==============================")
    print("🔍 INICIANDO MODEL SEARCH")
    print("==============================")

    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    import optuna

    best_overall = {
        "model_name": None,
        "model": None,
        "params": None,
        "threshold": 0.5,
        "f1": -1.0,
    }

    # ---------------------------------------------
    # 1) GRID SEARCH RÁPIDO
    # ---------------------------------------------
    print("\n➡️  GRID SEARCH (Warm Start)")

    experiments = [
        (
            "DecisionTree",
            DecisionTreeClassifier,
            [
                {"max_depth": 8, "min_samples_leaf": 1},
                {"max_depth": 12, "min_samples_leaf": 3},
                {"max_depth": 16, "min_samples_leaf": 7},
            ],
        ),
        (
            "RandomForest",
            RandomForestClassifier,
            [
                {"n_estimators": 50, "max_depth": 12, "min_samples_leaf": 2},
                {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 1},
                {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 2},
            ],
        ),
        (
            "LogisticRegression",
            LogisticRegression,
            [
                {
                    "C": 0.01,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 500,
                },
                {
                    "C": 0.1,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 500,
                },
                {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 500,
                },
            ],
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier,
            [
                {
                    "n_estimators": 50,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                },
                {
                    "n_estimators": 100,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                },
                {
                    "n_estimators": 150,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                },
            ],
        ),
    ]

    for name, model_class, grid in experiments:
        print(f"\n🧪 Probando modelo: {name}")

        for params in grid:
            params_local = dict(params)
            params_local["random_state"] = random_state

            print(f"   → Params: {params_local}")

            try:
                model = model_class(**params_local)
                model.fit(X_train, y_train)

                proba = model.predict_proba(X_val)[:, 1]
                pred = model.predict(X_val)
                f1_local = f1_score(y_val, pred, average="macro")
                t = 0.5
                print(f"      ✓ F1={f1_local:.4f}  | threshold={t:.3f}")

                if f1_local > best_overall["f1"]:
                    print("      ⭐ Nuevo mejor modelo encontrado")
                    best_overall.update(
                        {
                            "model_name": name,
                            "model": model,
                            "params": params_local,
                            "threshold": t,
                            "f1": f1_local,
                        }
                    )
            except Exception as e:
                print(f"      ❌ Error entrenando {name}: {str(e)}")

    # ---------------------------------------------
    # 2) OPTUNA (solo para el mejor modelo)
    # ---------------------------------------------
    print("\n➡️  OPTUNA (Refinamiento fino)")

    def objective(trial):
        model_name = best_overall["model_name"] or "DecisionTree"

        if model_name == "DecisionTree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 25),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 30
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 20
                ),
                "random_state": random_state,
            }
            model = DecisionTreeClassifier(**params)

        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 10
                ),
                "random_state": random_state,
            }
            model = RandomForestClassifier(**params)

        elif model_name == "GradientBoosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3
                ),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "random_state": random_state,
            }
            model = GradientBoostingClassifier(**params)

        elif model_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 0.001, 3.0),
                "max_iter": 800,
                "solver": "lbfgs",
            }
            model = LogisticRegression(**params)

        else:
            # Modelo inesperado → penalizamos
            return -1.0

        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return f1_score(y_val, pred, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trials)

    print("Best Params:", study.best_params)
    print(f"Best F1: {study.best_value:.4f}")

    model_name = best_overall["model_name"]
    print(f"\n🏆 Modelo ganador final: {model_name}")

    if model_name == "DecisionTree":
        model_final = DecisionTreeClassifier(
            **study.best_params, random_state=random_state
        )
    elif model_name == "RandomForest":
        model_final = RandomForestClassifier(
            **study.best_params, random_state=random_state
        )
    elif model_name == "GradientBoosting":
        model_final = GradientBoostingClassifier(
            **study.best_params, random_state=random_state
        )
    elif model_name == "LogisticRegression":
        model_final = LogisticRegression(**study.best_params)
    else:
        raise RuntimeError(f"Modelo inesperado en Optuna: {model_name}")

    model_final.fit(X_train, y_train)
    pred_final = model_final.predict(X_val)
    f1_final = f1_score(y_val, pred_final, average="macro")
    t_final = 0.5

    best_overall.update(
        {
            "model_name": model_name,
            "model": model_final,
            "params": study.best_params,
            "threshold": t_final,
            "f1": f1_final,
        }
    )

    print(f"\n🎯 Modelo final posterior a Optuna:")
    print(f"    → Modelo: {model_name}")
    print(f"    → Hyperparams: {study.best_params}")
    print(f"    → F1 Final: {f1_final:.4f}")
    print(f"    → Threshold óptimo: {t_final:.3f}")
    print("\n==============================")
    print("🏁 MODEL SEARCH COMPLETO")
    print("==============================")

    return (
        best_overall["model"],
        best_overall["params"],
        best_overall["model_name"],
        best_overall["threshold"],
    )