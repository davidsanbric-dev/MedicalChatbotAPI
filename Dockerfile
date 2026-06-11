# 🐍 Imagen base ligera con Python 3.12
FROM python:3.12-slim AS base

# ⚙️ Variables de entorno básicas
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_SYSTEM_PYTHON=1

# 📂 Directorio de trabajo
WORKDIR /app

# 🧱 Instala dependencias del sistema necesarias para compilar extensiones
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ⚡ Instala uv (el gestor universal ultrarrápido)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

# 🧩 Copia solo los archivos de configuración para aprovechar el cache de Docker
COPY pyproject.toml uv.lock ./
COPY README.md ./

# 📦 Copia el código fuente del proyecto (necesario antes de uv sync para builds editables)
COPY . .

# 🚀 Instala dependencias del proyecto dentro del sistema (sin venv)
RUN uv sync --frozen

# 🌍 Puerto configurable de la app (override con la env API_PORT en Dokploy/compose)
ENV API_PORT=8000
EXPOSE ${API_PORT}

# 🧠 Ejecuta la app con uvicorn usando uv (más rápido y reproducible).
# Forma shell + exec para que ${API_PORT} se expanda en runtime y uvicorn
# reciba las señales (SIGTERM) directamente del init del contenedor.
CMD ["sh", "-c", "exec uv run uvicorn app.main:app --host 0.0.0.0 --port ${API_PORT:-8000}"]
