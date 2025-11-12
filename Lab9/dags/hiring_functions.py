#Inserte su código aqui
import os 
def create_folders(**kwargs):
    """
    Crea una carpeta principal con el nombre de la fecha de ejecución (ds)
    y las subcarpetas 'raw', 'splits' y 'models'.
    """
    execution_date = kwargs.get('ds') 
    base_path = os.path.join(os.getcwd(), execution_date)
    os.makedirs(base_path, exist_ok=True)

    subfolders = ['raw', 'splits', 'models']
    for sub in subfolders:
        os.makedirs(os.path.join(base_path, sub), exist_ok=True)

    print(f"Carpetas creadas en: {base_path}")
    return base_path

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(base_path, seed=6):
    """
    Lee el dataset data_1.csv desde la carpeta 'raw',
    realiza un hold-out estratificado (80/20) y guarda
    los datasets en 'splits'.
    """
    raw_path = os.path.join(base_path, "raw", "data_1.csv")
    df = pd.read_csv(raw_path)

    # Separar variables
    X = df.drop(columns=["HiringDecision"])
    y = df["HiringDecision"]

    # Hold-out estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Guardar resultados
    split_path = os.path.join(base_path, "splits")
    X_train.to_csv(os.path.join(split_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(split_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(split_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(split_path, "y_test.csv"), index=False)

    print(f"Datos divididos y guardados en {split_path}")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

def preprocess_and_train(base_path):
    """
    Lee los sets de entrenamiento y prueba, crea un pipeline con preprocesamiento
    y entrena un modelo RandomForest. Guarda el pipeline entrenado y muestra métricas.
    """
    split_path = os.path.join(base_path, "splits")
    model_path = os.path.join(base_path, "models")

    # Cargar datos
    X_train = pd.read_csv(os.path.join(split_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(split_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(split_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(split_path, "y_test.csv")).values.ravel()

    # --- Definir columnas ---
    numeric_features = [
        "Age", "ExperienceYears", "PreviousCompanies",
        "DistanceFromCompany", "InterviewScore",
        "SkillScore", "PersonalityScore"
    ]
    categorical_features = ["Gender", "EducationLevel", "RecruitmentStrategy"]

    # --- Preprocesamiento ---
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # --- Modelo ---
    clf = RandomForestClassifier(
        n_estimators=150,
        random_state=6,
        n_jobs=-1
    )

    # --- Pipeline completo ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    # --- Entrenar ---
    pipeline.fit(X_train, y_train)

    # --- Evaluar ---
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy: {acc:.3f}")
    print(f"F1-score (Contratado=1): {f1_pos:.3f}")

    # --- Guardar modelo ---
    joblib.dump(pipeline, os.path.join(model_path, "hiring_model.joblib"))
    print(f"Modelo guardado en {model_path}/hiring_model.joblib")

import gradio as gr
def predict(file,model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}


def gradio_interface():
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    model_path = os.path.join(os.getcwd(), today, "models", "hiring_model.joblib")

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)
