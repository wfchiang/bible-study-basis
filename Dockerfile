# Single stage build for combined UI and MCP
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY vector_store_data ./vector_store_data
COPY py ./py
COPY config.yaml setup_env.sh start-ui.sh ./

RUN dos2unix setup_env.sh start-ui.sh && \
    chmod +x setup_env.sh start-ui.sh

# Expose 8080 for the UI. Ensure MCP uses a different internal port (e.g., 8081)
EXPOSE 8080
CMD ["./start-ui.sh"]