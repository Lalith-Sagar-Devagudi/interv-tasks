FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# poppler-utils needed by pdf2image; build-essential for compiled wheels
RUN apt-get update && apt-get install -y build-essential poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies first (layer-cached until pyproject.toml changes)
COPY backend/pyproject.toml backend/uv.lock* ./
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# Copy application source
COPY backend/app ./app

# uv creates a venv at /app/.venv — put it on PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
