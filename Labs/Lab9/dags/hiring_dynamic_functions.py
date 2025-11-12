import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def create_folders(**kwargs):
    """
    Crea una carpeta principal con el nombre de la fecha de ejecución (ds)
    y las subcarpetas: raw, preprocessed, splits y models.
    """
    execution_date = kwargs.get('ds', datetime.now().date().isoformat())
    base_path = os.path.join(os.getcwd(), execution_date)
    os.makedirs(base_path, exist_ok=True)

    subfolders = ['raw', 'preprocessed', 'splits', 'models']
    for sub in subfolders:
        os.makedirs(os.path.join(base_path, sub), exist_ok=True)

    print(f"Carpetas creadas en: {base_path}")
    return base_path


def load_and_merge(base_path):
    """
    Lee data_1.csv y opcionalmente data_2.csv desde 'raw',
    concatena y guarda el resultado en 'preprocessed'.
    """
    raw_path = os.path.join(base_path, 'raw')
    pre_path = os.path.join(base_path, 'preprocessed')

    files = [os.path.join(raw_path, 'data_1.csv')]
    data_2_path = os.path.join(raw_path, 'data_2.csv')
    if os.path.exists(data_2_path):
        files.append(data_2_path)

    dfs = [pd.read_csv(f) for f in files]
    df_merged = pd.concat(dfs, ignore_index=True)

    merged_path = os.path.join(pre_path, 'merged_data.csv')
    df_merged.to_csv(merged_path, index=False)
    print(f"Datos combinados guardados en: {merged_path}")


def split_data(base_path, seed=42):
    """
    Lee merged_data.csv desde 'preprocessed', aplica un hold-out 80/20,
    y guarda los conjuntos en 'splits'.
    """
    pre_path = os.path.join(base_path, 'preprocessed')
    df = pd.read_csv(os.path.join(pre_path, 'merged_data.csv'))

    X = df.drop(columns=["HiringDecision"])
    y = df["HiringDecision"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    split_path = os.path.join(base_path, 'splits')
    X_train.to_csv(os.path.join(split_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(split_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(split_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(split_path, "y_test.csv"), index=False)

    print(f"Datos divididos y guardados en {split_path}")


def train_model(base_path, model, model_name):
    """
    Entrena un modelo recibido como parámetro usando un pipeline con preprocesamiento.
    Guarda el pipeline entrenado en 'models' como <model_name>.joblib.
    """
    split_path = os.path.join(base_path, "splits")
    model_path = os.path.join(base_path, "models")

    X_train = pd.read_csv(os.path.join(split_path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(split_path, "y_train.csv")).values.ravel()

    # Columnas
    numeric_features = [
        "Age", "ExperienceYears", "PreviousCompanies",
        "DistanceFromCompany", "InterviewScore", "SkillScore", "PersonalityScore"
    ]
    categorical_features = ["Gender", "EducationLevel", "RecruitmentStrategy"]

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

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    save_path = os.path.join(model_path, f"{model_name}.joblib")
    joblib.dump(pipeline, save_path)
    print(f"Modelo '{model_name}' entrenado y guardado en {save_path}")



def evaluate_models(base_path):
    """
    Evalúa todos los modelos entrenados en 'models' y selecciona el de mejor accuracy.
    Guarda ese modelo como 'best_model.joblib'.
    """
    split_path = os.path.join(base_path, "splits")
    model_path = os.path.join(base_path, "models")

    X_test = pd.read_csv(os.path.join(split_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(split_path, "y_test.csv")).values.ravel()

    results = {}

    for file in os.listdir(model_path):
        if file.endswith(".joblib") and file != "best_model.joblib":
            model_file = os.path.join(model_path, file)
            model_name = file.replace(".joblib", "")
            pipeline = joblib.load(model_file)

            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[model_name] = acc
            print(f"Modelo: {model_name} | Accuracy: {acc:.3f}")

    if not results:
        raise ValueError("No se encontraron modelos entrenados en la carpeta 'models'.")

    best_model_name = max(results, key=results.get)
    best_acc = results[best_model_name]

    best_model_path = os.path.join(model_path, f"{best_model_name}.joblib")
    best_copy_path = os.path.join(model_path, "best_model.joblib")
    joblib.dump(joblib.load(best_model_path), best_copy_path)

    print(f"Mejor modelo: {best_model_name} | Accuracy: {best_acc:.3f}")
    print(f"Guardado como {best_copy_path}")