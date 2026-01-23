# =============================================
# SenseVoice API 服务容器镜像
# =============================================
# 构建命令:
#   docker build -t sensevoice-api:latest .
#
# 推送命令:
#   docker tag sensevoice-api:latest ${REGISTRY}/sensevoice-api:latest
#   docker push ${REGISTRY}/sensevoice-api:latest
#
# 环境变量说明:
#   MODEL_NAME        - 模型名称，默认 iic/SenseVoiceSmall
#   SENSEVOICE_DEVICE - 设备类型 (cuda/cpu)
#   PRELOAD_MODEL     - 是否预加载模型 (1=启动时加载, 0=首次请求时加载)
# =============================================

# 镜像版本信息（可自定义）
ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.1
ARG CUDNN_VERSION=8

# =============================================
# 阶段1: 构建依赖 (builder)
# =============================================
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并构建 wheel 缓存
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# =============================================
# 阶段2: 运行镜像 (runtime)
# =============================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu22.04

LABEL maintainer="SenseVoice" \
      description="SenseVoice API Service - OpenAI Compatible Speech-to-Text" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/FunAudioLLM/SenseVoice"

# 安装系统运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制 wheels
COPY --from=builder /app/wheels /app/wheels

# 安装 Python 依赖（使用预编译的 wheels 以加速构建）
RUN pip install --no-cache-dir --find-links /app/wheels -r /app/wheels/../requirements.txt && \
    rm -rf /app/wheels

# 安装 FastAPI 运行时依赖
RUN pip install --no-cache-dir \
    fastapi>=0.111.1 \
    uvicorn>=0.27.0 \
    python-multipart>=0.0.6 \
    aiofiles>=23.2.1 \
    prometheus-client>=0.20.0

# 设置工作目录
WORKDIR /app
COPY . .

# 创建必要目录
RUN mkdir -p /tmp/sensevoice /app/models /app/logs && chmod -R 777 /tmp/sensevoice /app/models /app/logs

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM="false" \
    CUDA_MODULE_LOADING=LAZY \
    CUDA_LAUNCH_BLOCKING=0 \
    MODELSCOPE_CACHE="/app/models" \
    HF_HOME="/tmp/.cache" \
    LOG_DIR="/app/logs" \
    SENSEVOICE_DEVICE="cuda"

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# 启动入口
ENTRYPOINT ["python", "openai_api.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "1"]