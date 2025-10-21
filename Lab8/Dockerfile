# Imagen base de Python
FROM python:3.12-slim

# Definir directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos necesarios al contenedor
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models/ ./models

# Exponer el puerto donde correrá la API
EXPOSE 8000

# 5️⃣ Comando de ejecución
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]