FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Build tools for any dependency without a prebuilt wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

# The project installs itself (editable), so its code is copied before uv sync
COPY . .
RUN uv sync --frozen --no-dev

# Port uvicorn binds; override with API_PORT (Dokploy or compose)
ENV API_PORT=8000
EXPOSE ${API_PORT}

# Shell form so ${API_PORT} expands at runtime; exec so uvicorn receives SIGTERM directly
CMD ["sh", "-c", "exec uv run --no-dev uvicorn app.main:app --host 0.0.0.0 --port ${API_PORT:-8000}"]
