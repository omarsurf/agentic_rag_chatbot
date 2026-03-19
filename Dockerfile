# =============================================================================
# GRI RAG API - Production Dockerfile
# =============================================================================
# Multi-stage build for minimal image size

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

# Security: Create non-root user
RUN groupadd -r gri && useradd -r -g gri gri

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create data directories with proper permissions
RUN mkdir -p /app/data /app/reports && \
    chown -R gri:gri /app

# Switch to non-root user
USER gri

# Environment defaults (can be overridden)
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_RELOAD=false \
    API_WORKERS=4 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose port
EXPOSE 8000

# Entry point with uvicorn for production
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4"]
