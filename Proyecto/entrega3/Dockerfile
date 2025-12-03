FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar requirements (se instalarán a continuación)
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    libgomp1 \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt \
    && apt-get remove -y build-essential gcc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copiar el repositorio (scripts de predicción y utilidades)
COPY . /app

RUN chmod +x /app/Proyecto/entrega3/entrypoint.sh || true

# Por defecto el entrypoint lanzará la predicción para 2025-01-01 - 2025-01-05
ENTRYPOINT ["/app/Proyecto/entrega3/entrypoint.sh"]
