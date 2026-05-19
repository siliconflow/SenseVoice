# ==================== SenseVoice API Server ====================
# Optimized Dockerfile - Multi-stage build for smaller image
# Build target: linux/amd64
# Supports: RTX 4090 (Ada) and RTX 5090 (Blackwell)
#
# Build args:
#   TORCH_VERSION: PyTorch version to install (default: 2.5.1 for RTX 4090, 2.6.0 for RTX 5090)
#   TORCH_INDEX_URL: PyTorch index URL (default: cu124 for RTX 4090, cu128 for RTX 5090)

# ==================== Stage 1: Build dependencies ====================
FROM python:3.10-slim-bookworm AS builder

# Build arguments for PyTorch version selection
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

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

# Install PyTorch with specific version and CUDA support
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchaudio --index-url ${TORCH_INDEX_URL}

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==================== Stage 2: Runtime image ====================
FROM python:3.10-slim-bookworm AS runtime

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
COPY model.py .
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
    HF_HOME=/models \
    SENSEVOICE_DEVICE="cuda" \
    API_PORT=80 \
    GRACEFUL_SHUTDOWN_TIMEOUT="30"

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:80/health || exit 1

# Default command
CMD ["python", "api.py"]