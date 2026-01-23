# ==================== SenseVoice API Server ====================
# Optimized Dockerfile - Multi-stage build for smaller image
# Build target: linux/amd64

# ==================== Stage 1: Build dependencies ====================
FROM mirror.gcr.io/library/python:3.10-slim-bookworm AS builder

# Use Aliyun mirror for apt
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies with Aliyun pip mirror
COPY requirements.txt .
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== Stage 2: Runtime image ====================
FROM mirror.gcr.io/library/python:3.10-slim-bookworm AS runtime

# Use Aliyun mirror for apt
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true

# Install runtime dependencies (no CUDA, provided by runtime environment)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY api.py .
COPY utils/ ./utils/

# Create non-root user
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /models /tmp/sensevoice /app/logs && \
    chown -R appuser:appuser /models /tmp/sensevoice /app/logs

USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    TEMP_DIR=/tmp/sensevoice \
    LOG_DIR=/app/logs \
    MODELSCOPE_CACHE=/models \
    HF_HOME=/tmp/.cache \
    SENSEVOICE_DEVICE="cuda"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Default command
CMD ["python", "api.py"]