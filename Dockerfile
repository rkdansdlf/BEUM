# ==============================================================================
# Production Dockerfile for BEUM Central Receiver Server & DrainSight Dashboard
# ==============================================================================
FROM python:3.12-slim AS runtime

# Install system dependencies (curl for healthcheck, sqlite3 for database management)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Optimize Python execution in container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RECEIVER_HOST=0.0.0.0 \
    RECEIVER_PORT=8001 \
    RECEIVER_DATA_DIR=/app/received_data

WORKDIR /app

# Create non-root system user and prepare volume directory
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser && \
    mkdir -p /app/received_data && \
    chown -R appuser:appgroup /app

# Install minimal server Python dependencies
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy server application source code and migrations
COPY receiver_server.py drainsight_adapter.py schemas.py ./
COPY migrations/ ./migrations/

# Fix ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose default server port
EXPOSE 8001

# Healthcheck against FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${RECEIVER_PORT}/health || exit 1

# Launch receiver server
ENTRYPOINT ["python", "receiver_server.py"]
CMD ["--host", "0.0.0.0", "--port", "8001", "--data-dir", "/app/received_data"]
