from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import os


from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface


default_args = {
    'owner': 'matias',
    'start_date': datetime(2024, 10, 1),
    'retries': 0
}

with DAG(
    dag_id='hiring_lineal',
    default_args=default_args,
    schedule_interval=None,   
    catchup=False,            
    tags=['lab9', 'mlops']
) as dag:


    start = EmptyOperator(task_id='inicio_pipeline')

    create_folders_task = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        provide_context=True,  
    )

    def download_data(**kwargs):
        execution_date = kwargs.get('ds')
        base_path = os.path.join(os.getcwd(), execution_date)
        raw_path = os.path.join(base_path, 'raw')
        os.makedirs(raw_path, exist_ok=True)

        url = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        output_path = os.path.join(raw_path, "data_1.csv")

        os.system(f"curl -s -o {output_path} {url}")

        if os.path.exists(output_path):
            print(f"✅ Archivo descargado correctamente en {output_path}")
        else:
            raise FileNotFoundError("❌ Error: No se descargó el archivo data_1.csv")

    download_data_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
        provide_context=True,
    )


    def run_split(**kwargs):
        base_path = os.path.join(os.getcwd(), kwargs.get('ds'))
        split_data(base_path)

    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=run_split,
        provide_context=True,
    )

    def run_train(**kwargs):
        base_path = os.path.join(os.getcwd(), kwargs.get('ds'))
        preprocess_and_train(base_path)

    preprocess_task = PythonOperator(
        task_id='preprocess_and_train',
        python_callable=run_train,
        provide_context=True,
    )


    gradio_task = PythonOperator(
        task_id='gradio_interface',
        python_callable=gradio_interface,
    )


    start >> create_folders_task >> download_data_task >> split_data_task >> preprocess_task >> gradio_task