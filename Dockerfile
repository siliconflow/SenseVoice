# SenseVoice API 服务容器镜像 - 支持 NVIDIA GPU、OpenAI 兼容 API
# 编译命令: docker build -t sensevoice-api .
# 运行命令: docker run -d --gpus all -p 8000:8000 sensevoice-api

# ============== 构建阶段 ==============
FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu22.04 as builder

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    pip \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装 PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装额外依赖（API 服务）
RUN pip install --no-cache-dir \
    fastapi>=0.111.1 \
    uvicorn>=0.27.0 \
    python-multipart>=0.0.6 \
    aiofiles>=23.2.1 \
    prometheus-client>=0.20.0

# ============== 运行阶段 ==============
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="SenseVoice" \
      description="SenseVoice API Service - OpenAI Compatible Speech-to-Text"

WORKDIR /workspace

# 安装系统运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 设置 Python 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    CUDA_MODULE_LOADING=LAZY \
    CUDA_LAUNCH_BLOCKING=0

# 设置工作目录
WORKDIR /app
COPY . /app

# 创建必要目录
RUN mkdir -p /tmp/sensevoice /app/models /app/logs && chmod -R 777 /tmp/sensevoice /app/models /app/logs

# 设置环境变量
ENV SENSEVOICE_DEVICE="cuda" \
    TOKENIZERS_PARALLELISM="false" \
    MODELSCOPE_CACHE="/app/models" \
    LOG_DIR="/app/logs"

# 暴露端口（Prometheus metrics 与主 API 同端口，通过 /metrics 路径暴露）
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# 启动入口
ENTRYPOINT ["python", "openai_api.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "1"]