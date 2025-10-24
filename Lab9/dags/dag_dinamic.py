from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import os

from hiring_dynamic_functions import (
    create_folders,
    load_and_merge,
    split_data,
    train_model,
    evaluate_models
)


default_args = {
    'owner': 'matias',
    'start_date': datetime(2024, 10, 1),
    'retries': 0
}

with DAG(
    dag_id='hiring_dynamic',
    default_args=default_args,
    schedule_interval='0 15 5 * *',  
    catchup=True,                    
    tags=['lab9', 'mlops', 'dynamic']
) as dag:

    start = EmptyOperator(task_id='inicio_pipeline')

    create_folders_task = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        provide_context=True,
    )


    def choose_branch(**kwargs):
        execution_date = datetime.strptime(kwargs['ds'], "%Y-%m-%d")
        cutoff = datetime(2024, 11, 1)

        if execution_date < cutoff:
            return 'download_data1'
        else:
            return 'download_data1_and_2'

    branching = BranchPythonOperator(
        task_id='branching_download_logic',
        python_callable=choose_branch,
        provide_context=True,
    )


    def download_data1(**kwargs):
        execution_date = kwargs.get('ds')
        base_path = os.path.join(os.getcwd(), execution_date, 'raw')
        os.makedirs(base_path, exist_ok=True)

        url = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        os.system(f"curl -s -o {os.path.join(base_path, 'data_1.csv')} {url}")
        print("✅ data_1.csv descargado correctamente")

    def download_data1_and_2(**kwargs):
        execution_date = kwargs.get('ds')
        base_path = os.path.join(os.getcwd(), execution_date, 'raw')
        os.makedirs(base_path, exist_ok=True)

        urls = [
            ("data_1.csv", "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"),
            ("data_2.csv", "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv")
        ]
        for fname, url in urls:
            os.system(f"curl -s -o {os.path.join(base_path, fname)} {url}")
            print(f"✅ {fname} descargado correctamente")

    download_data1_task = PythonOperator(
        task_id='download_data1',
        python_callable=download_data1,
        provide_context=True,
    )

    download_data1_and_2_task = PythonOperator(
        task_id='download_data1_and_2',
        python_callable=download_data1_and_2,
        provide_context=True,
    )


    def run_load_and_merge(**kwargs):
        base_path = os.path.join(os.getcwd(), kwargs.get('ds'))
        load_and_merge(base_path)

    load_and_merge_task = PythonOperator(
        task_id='load_and_merge',
        python_callable=run_load_and_merge,
        provide_context=True,
        trigger_rule=TriggerRule.ONE_SUCCESS, 
    )


    def run_split(**kwargs):
        base_path = os.path.join(os.getcwd(), kwargs.get('ds'))
        split_data(base_path)

    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=run_split,
        provide_context=True,
    )


    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    def train_rf(**kwargs):
        base_path = os.path.join(os.getcwd(), kwargs.get('ds'))
        model = RandomForestClassifier(n_estimators=150, random_state=6)
        train_model(base_path, model, "RandomForest")

    def train_gb(**kwargs):
        base_path = os.path.join(os.getcwd(), kwargs.get('ds'))
        model = GradientBoostingClassifier(random_state=6)
        train_model(base_path, model, "GradientBoosting")

    def train_lr(**kwargs):
        base_path = os.path.join(os.getcwd(), kwargs.get('ds'))
        model = LogisticRegression(max_iter=500)
        train_model(base_path, model, "LogisticRegression")

    train_rf_task = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_rf,
        provide_context=True,
    )

    train_gb_task = PythonOperator(
        task_id='train_gradient_boosting',
        python_callable=train_gb,
        provide_context=True,
    )

    train_lr_task = PythonOperator(
        task_id='train_logistic_regression',
        python_callable=train_lr,
        provide_context=True,
    )


    def run_evaluate(**kwargs):
        base_path = os.path.join(os.getcwd(), kwargs.get('ds'))
        evaluate_models(base_path)

    evaluate_task = PythonOperator(
        task_id='evaluate_models',
        python_callable=run_evaluate,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,  
    )

    start >> create_folders_task >> branching
    branching >> [download_data1_task, download_data1_and_2_task] >> load_and_merge_task
    load_and_merge_task >> split_data_task >> [train_rf_task, train_gb_task, train_lr_task] >> evaluate_task