# ==================== SenseVoice API Server ====================
# 优化版 Dockerfile - 使用多阶段构建减小体积
# 构建目标: linux/amd64

# ==================== 阶段1: 构建依赖 ====================
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

# 安装 build 依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== 阶段2: 运行镜像 ====================
FROM python:3.10-slim-bookworm AS runtime

# 安装运行时依赖 (不含 CUDA，由运行时环境提供)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY api.py .
COPY utils/ ./utils/

# 创建非 root 用户
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /models /tmp/sensevoice /app/logs && \
    chown -R appuser:appuser /models /tmp/sensevoice /app/logs

USER appuser

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    TEMP_DIR=/tmp/sensevoice \
    LOG_DIR=/app/logs \
    MODELSCOPE_CACHE=/models \
    HF_HOME=/tmp/.cache \
    SENSEVOICE_DEVICE="cuda"

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# 默认启动命令
CMD ["python", "api.py"]