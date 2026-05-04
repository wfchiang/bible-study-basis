# Single stage build for combined UI and MCP
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

COPY data ./data
COPY vector_store_data ./vector_store_data
COPY config.yaml start.sh ./

RUN dos2unix start.sh && chmod +x start.sh

# Expose 8080 for the UI. Ensure MCP uses a different internal port (e.g., 8081)
EXPOSE 8080
CMD ["./start.sh"]