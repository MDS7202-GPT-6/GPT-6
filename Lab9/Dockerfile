# Imagen base ligera
FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Definir el directorio de trabajo
WORKDIR /opt/airflow

# Crear directorios requeridos por Airflow
RUN mkdir -p dags logs plugins

# Copiar los requerimientos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar DAGs al contenedor
COPY dags/ dags/

# Definir variable de entorno de Airflow
ENV AIRFLOW_HOME=/opt/airflow

# Comando por defecto: iniciar Airflow standalone
CMD ["airflow", "standalone"]