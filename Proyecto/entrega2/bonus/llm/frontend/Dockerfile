FROM python:3.11-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY chat.py .

# Exponer puerto
EXPOSE 7862

# Comando para iniciar
CMD ["python", "chat.py"]
