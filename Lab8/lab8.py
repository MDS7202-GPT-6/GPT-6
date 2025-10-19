import os
import pickle
import json
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import plotly.io as pio
from datetime import datetime
import importlib.metadata as md
import subprocess


# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def load_data():
    df = pd.read_csv("/Users/matiasgodoy/Universidad/2025-2/Lab MDS/GPT-6/Lab8/water_potability.csv")
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    return train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)


def create_preprocessor(X):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), X.columns)
    ])


def suggest_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2)
    }


def create_model(preprocessor, params):
    return Pipeline([
        ("pre", preprocessor),
        ("xgb", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=6,
            n_jobs=-1,
            **params
        ))
    ])


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    best_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_id + "/model")
    return best_model




def log_requirements():
    """Genera un requirements.txt con todas las dependencias y lo sube a MLflow."""
    req_path = "requirements.txt"
    with open(req_path, "w") as f:
        subprocess.run(["pip", "freeze"], stdout=f, text=True, check=True)
    mlflow.log_artifact(req_path)



def optimize_model():
    # --- Configurar tracking local ---
    tracking_dir = "./mlruns"
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
    mlflow.end_run()  # Cierra cualquier run activo previo

    # --- Cargar datos ---
    X_train, X_valid, y_train, y_valid = load_data()
    preprocessor = create_preprocessor(X_train)

    # --- Crear experimento nuevo ---
    exp_name = f"Lab8_XGBoost_Primer_Exp"
    mlflow.set_experiment(exp_name)
    experiment = mlflow.get_experiment_by_name(exp_name)
    exp_id = experiment.experiment_id

    # --- Función objetivo (Optuna) ---
    def objective(trial):
        params = suggest_params(trial)
        model = create_model(preprocessor, params)
        run_name = run_name = f"XGB lr={params['learning_rate']:.3f} nest={params['n_estimators']} depth={params['max_depth']} mcw={params['min_child_weight']:.1f} subs={params['subsample']:.2f} col={params['colsample_bytree']:.2f} g={params['gamma']:.2f} a={params['reg_alpha']:.2f} l={params['reg_lambda']:.2f}"

        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            mlflow.sklearn.autolog(log_models=True, silent=True)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            f1 = f1_score(y_valid, preds)
            mlflow.log_metric("valid_f1", f1)
            return f1

    # --- Ejecutar optimización ---
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=6))
    study.optimize(objective, n_trials=20)

    # --- Crear carpetas de artefactos ---
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # --- Guardar gráficos de Optuna ---
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    pio.write_html(fig1, file="plots/history.html", auto_open=False)
    pio.write_html(fig2, file="plots/importances.html", auto_open=False)
    mlflow.log_artifact("plots/history.html", artifact_path="plots")
    mlflow.log_artifact("plots/importances.html", artifact_path="plots")

    # --- Guardar mejores parámetros ---
    with open("plots/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    mlflow.log_artifact("plots/best_params.json", artifact_path="plots")

    # --- Guardar mejor modelo ---
    best_model = get_best_model(exp_id)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact("models/best_model.pkl", artifact_path="models")

    # --- Importancia de características ---
    model_final = best_model.named_steps["xgb"]
    importances = model_final.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.bar(X_train.columns, importances)
    plt.xticks(rotation=45)
    plt.title("Importancia de características")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    mlflow.log_artifact("plots/feature_importance.png", artifact_path="plots")

    # --- Guardar versiones de librerías ---
    log_requirements()

    print(f"Experimento: {exp_name}")
    print(f"Mejor F1-score: {study.best_value:.4f}")
    print(f"Modelo guardado en: models/best_model.pkl")
    return exp_id, study


optimize_model()