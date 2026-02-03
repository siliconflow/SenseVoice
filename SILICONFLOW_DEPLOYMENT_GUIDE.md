# SiliconFlow 平台部署手册

## 概述

本手册总结 SenseVoice 项目部署到 SiliconFlow 平台的经验，提供标准化的部署流程和检查清单，帮助其他项目快速满足 SiliconFlow 平台的部署要求。

**内容结构：**
1. **普适性经验** - 适用于任何 AI 服务部署
2. **语音领域经验** - 针对语音/音频服务的特殊需求

---

## 第一部分：普适性经验（适用于任何 AI 服务）

### 1.1 API 设计规范

#### 1.1.1 兼容行业标准协议

**为什么：** 兼容广泛使用的 API 规范可以大幅降低用户迁移成本，提高服务可用性。

| 场景 | 推荐协议 | 示例 |
|-----|---------|------|
| 文本生成 | OpenAI Chat Completions | `/v1/chat/completions` |
| 文本转语音 | OpenAI Audio API | `/v1/audio/speech` |
| 语音转文本 | OpenAI Transcriptions | `/v1/audio/transcriptions` |
| 向量嵌入 | OpenAI Embeddings | `/v1/embeddings` |
| 多模态 | Anthropic Messages | `/v1/messages` |

**实施要点：**
- 保持请求/响应格式与标准协议一致
- 支持相同的认证方式（Bearer Token）
- 提供兼容的模型名称映射

```python
# OpenAI 兼容接口示例
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 即使底层模型不同，也保持接口兼容
    return {
        "id": "chatcmpl-xxx",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [...]
    }
```

#### 1.1.2 健康检查接口设计（关键）

**为什么：** K8s 需要区分服务是否存活（Liveness）和是否可接收流量（Readiness）。**Readiness probe 必须切实有效，不能只是形式化检查，要真实反映实例的服务响应能力。**

**接口定义：**
- `/health`（Liveness）：简单快速，只检查进程是否活着
- `/ready`（Readiness）：**必须执行实际的业务逻辑测试**，验证服务真正能处理请求

```python
@app.get("/health")
async def health():
    """存活检查 - 简单快速，只检查进程是否活着"""
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    """
    就绪检查 - 必须真实反映服务响应能力

    检查项：
    1. 系统资源（内存、GPU）
    2. 模型加载状态
    3. 【关键】实际推理测试 - 确保模型能正常执行
    """
    errors = []

    # 1. 检查系统资源
    mem = psutil.virtual_memory()
    if mem.percent > 90:
        errors.append(f"high_memory_usage")

    # 2. 检查模型加载状态
    if not model_loaded:
        errors.append("model_not_ready")

    # 3. 【关键】执行实际推理测试
    if should_perform_inference_test():
        try:
            result = perform_actual_inference_test()
            if result.failed:
                errors.append(f"inference_test_failed")
        except Exception as e:
            errors.append(f"inference_test_error: {str(e)}")

    if errors:
        return {"status": "not_ready", "errors": errors}, 503

    return {"status": "ready"}
```

**形式化检查 vs 实质性检查：**

| 检查项 | 形式化检查（无效） | 实质性检查（有效） |
|-------|------------------|------------------|
| 模型状态 | 检查变量 `model_loaded = True` | 执行实际推理，验证模型能正常输出 |
| 数据库 | 检查连接对象存在 | 执行 `SELECT 1` 验证连接可用 |
| 缓存 | 检查客户端初始化 | 执行 `ping` 或 `get/set` 测试 |
| 外部服务 | 检查配置存在 | 发送真实探测请求验证连通性 |

**推理测试实现示例：**

```python
def perform_actual_inference_test():
    """
    执行实际的模型推理测试
    这是确保服务真正可用的关键
    """
    try:
        # 使用测试数据执行完整推理流程
        test_input = load_test_sample()  # 加载测试样本

        # 执行推理（与真实请求相同的代码路径）
        result = model.inference(test_input)

        # 验证输出有效性
        if not result or len(result) == 0:
            return InferenceResult(failed=True, error="empty_output")

        return InferenceResult(failed=False, output=result)

    except CUDAOutOfMemoryError as e:
        # GPU 内存不足，服务暂时不可用
        return InferenceResult(failed=True, error="cuda_oom")
    except Exception as e:
        # 其他推理错误
        return InferenceResult(failed=True, error=str(e))
```

**Readiness Probe 配置建议：**

```yaml
# deployment.yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 60      # 给模型加载预留时间
  periodSeconds: 30            # 检查频率
  timeoutSeconds: 30           # 推理测试可能需要较长时间
  failureThreshold: 3          # 连续失败3次才标记为未就绪
  successThreshold: 1

livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30
  timeoutSeconds: 5            # 健康检查应该很快
  failureThreshold: 3
```

### 1.2 K8s 优雅退出支持（关键）

#### 1.2.1 为什么需要优雅退出

当 K8s 需要缩容、滚动更新或重启 Pod 时，会发送 SIGTERM 信号给容器。如果没有正确处理：
- 正在处理的请求会被强制中断
- 用户收到错误响应
- 数据可能不一致或丢失

#### 1.2.2 优雅退出实现

```python
import signal
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 全局标志位，用于控制服务状态
_is_shutting_down = False
_active_requests = 0
_shutdown_event = asyncio.Event()
GRACEFUL_SHUTDOWN_TIMEOUT = int(os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT", "30"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("Service starting...")
    yield
    print("Service shutting down...")

app = FastAPI(lifespan=lifespan)

def handle_sigterm(signum, frame):
    """处理 SIGTERM 信号（K8s 发送）"""
    global _is_shutting_down
    print(f"Received SIGTERM, starting graceful shutdown...")
    _is_shutting_down = True

    # 设置事件，通知所有协程开始退出
    asyncio.create_task(wait_for_requests_complete())

async def wait_for_requests_complete():
    """等待所有活跃请求完成"""
    global _active_requests

    # 等待最多 30 秒（K8s terminationGracePeriodSeconds）
    max_wait = GRACEFUL_SHUTDOWN_TIMEOUT
    waited = 0

    while _active_requests > 0 and waited < max_wait:
        print(f"Waiting for {_active_requests} active requests to complete...")
        await asyncio.sleep(1)
        waited += 1

    print(f"Graceful shutdown complete. {_active_requests} requests remaining.")
    _shutdown_event.set()

    # 退出进程
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

@app.middleware("http")
async def shutdown_middleware(request, call_next):
    """中间件：检查服务是否正在关闭"""
    global _active_requests, _is_shutting_down

    # 如果正在关闭，拒绝新请求
    if _is_shutting_down:
        return JSONResponse(
            status_code=503,
            content={"error": "Service is shutting down"}
        )

    # 增加活跃请求计数
    _active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        # 减少活跃请求计数
        _active_requests -= 1
```

#### 1.2.3 K8s 配置

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      # 关键：设置优雅退出超时时间
      terminationGracePeriodSeconds: 35

      containers:
      - name: api
        # 关键：配置健康检查
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30

        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

        # 关键：配置优雅退出生命周期钩子
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 5"]
```

#### 1.2.4 优雅退出流程

```
K8s 发送 SIGTERM
    │
    ▼
┌─────────────────┐
│ 1. 设置关闭标志  │  ← 拒绝新请求
│ _is_shutting_down│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 2. 等待活跃请求  │  ← 继续处理进行中的请求
│    完成         │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 3. 超时或完成   │  ← 最多等待 terminationGracePeriodSeconds
│    退出进程     │
└─────────────────┘
```

### 1.3 容器化最佳实践

#### 1.3.1 镜像体积控制

**为什么：** 镜像体积直接影响部署速度和存储成本。

| 策略 | 方法 | 效果 |
|-----|------|------|
| 多阶段构建 | 分离编译和运行环境 | 减少 50-80% 体积 |
| 虚拟环境 | 使用 Python venv | 避免系统依赖污染 |
| 延迟下载 | 启动时下载模型权重 | 避免镜像包含大文件 |
| 精简基础镜像 | 使用 slim/alpine | 减少基础体积 |

**多阶段构建示例：**

```dockerfile
# Stage 1: 构建
FROM python:3.10-slim AS builder
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install -r requirements.txt

# Stage 2: 运行
FROM python:3.10-slim
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY . /app
WORKDIR /app
CMD ["python", "main.py"]
```

#### 1.3.2 安全实践

| 实践 | 原因 | 实现 |
|-----|------|------|
| 非 root 用户 | 防止容器逃逸攻击 | `USER appuser` |
| 只读文件系统 | 防止运行时修改 | `readOnlyRootFilesystem: true` |
| 资源限制 | 防止资源耗尽 | `resources.limits` |
| 健康检查 | 自动恢复故障 | `HEALTHCHECK` 指令 |

```dockerfile
# 创建非 root 用户
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser
```

### 1.4 日志与可观测性

#### 1.4.1 结构化日志

**为什么：** 便于日志收集系统解析和查询。

```python
import logging
import json

# 基础配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# 请求日志记录
def log_request(request, endpoint: str, extra: dict = None):
    """记录请求信息"""
    log_data = {
        "endpoint": endpoint,
        "client": request.client.host if request.client else "unknown",
        "trace_ids": {
            "x_trace_id": request.headers.get("X-Trace-Id"),
            "x_siliconcloud_trace_id": request.headers.get("x-siliconcloud-trace-id"),
        }
    }
    if extra:
        log_data.update(extra)

    logger.info(f"[{endpoint}] {json.dumps(log_data)}")
```

#### 1.4.2 关键指标记录

| 指标 | 用途 | 示例 |
|-----|------|------|
| 请求处理时间 | 性能监控 | `total: 1.234s, inference: 0.523s` |
| 资源使用 | 容量规划 | `gpu_memory: 8.5GB/24GB` |
| 错误类型 | 问题定位 | `error: CUDA_OUT_OF_MEMORY` |
| 队列长度 | 负载评估 | `queue_size: 5` |

#### 1.4.3 配置项环境变量化

**为什么：** 所有可配置项都应通过环境变量暴露，便于部署时调整而不需要重新构建镜像。

| 配置类别 | 环境变量 | 默认值 | 说明 |
|---------|---------|-------|------|
| **模型配置** | `MODEL_NAME` | - | 模型名称 |
| | `SENSEVOICE_DEVICE` | `cuda` | 运行设备 |
| | `ENABLE_MODEL_WARMUP` | `false` | 是否开启模型预热 |
| **性能配置** | `LARGE_FILE_THRESHOLD_MB` | `50` | 大文件阈值 |
| | `TEMP_FILE_CLEANUP_DELAY` | `300` | 临时文件清理延迟 |
| | `OMP_NUM_THREADS` | `4` | OpenMP 线程数 |
| **优雅退出** | `GRACEFUL_SHUTDOWN_TIMEOUT` | `30` | 优雅退出超时（秒）|

**实现示例：**

```python
import os

# 所有配置项都有默认值，同时允许环境变量覆盖
LARGE_FILE_THRESHOLD_MB = float(os.getenv("LARGE_FILE_THRESHOLD_MB", "50"))
TEMP_FILE_CLEANUP_DELAY = int(os.getenv("TEMP_FILE_CLEANUP_DELAY", "300"))
ENABLE_WARMUP = os.getenv("ENABLE_MODEL_WARMUP", "false").lower() == "true"
GRACEFUL_SHUTDOWN_TIMEOUT = int(os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT", "30"))

# 模型配置
model_kwargs = {
    "model": os.getenv("MODEL_NAME", "iic/SenseVoiceSmall"),
    "device": os.getenv("SENSEVOICE_DEVICE", "cuda:0"),
}

# 可选功能：仅当手工指定时才加载
punc_model = os.getenv("SENSEVOICE_PUNC_MODEL", "")
if punc_model:
    model_kwargs["punc_model"] = punc_model
```

### 1.5 模型加载策略

#### 1.5.1 启动时下载 vs 镜像内置

| 方案 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 启动时下载 | 镜像小，模型可更新 | 首次启动慢 | 模型经常更新 |
| 镜像内置 | 启动快，无网络依赖 | 镜像大，模型固定 | 模型稳定，追求启动速度 |
| 共享卷挂载 | 启动快，镜像小 | 需要预置卷 | K8s 环境 |

**启动时下载示例：**

```python
import os
from modelscope import snapshot_download

MODEL_NAME = os.getenv("MODEL_NAME", "default-model")
CACHE_DIR = os.getenv("MODELSCOPE_CACHE", "/models")

def download_model():
    """启动时下载模型"""
    model_path = snapshot_download(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        allow_file_pattern=['*.bin', '*.safetensors', '*.json']
    )
    return model_path

# 启动时执行
model_path = download_model()
model = load_model(model_path)
```

#### 1.5.2 模型预热

**为什么：** 避免第一次请求时冷启动延迟。

```python
# 启动时进行预热推理
def warmup_model(model):
    """模型预热"""
    import time
    start = time.time()

    # 使用虚拟输入进行推理
    dummy_input = torch.zeros(1, 3, 224, 224)
    _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Model warmup completed in {time.time() - start:.3f}s")

# 启动时调用
warmup_model(model)
```

### 1.6 并发与资源管理

#### 1.6.1 并发控制策略

| 策略 | 实现 | 适用场景 |
|-----|------|---------|
| 异步处理 | `async/await` | I/O 密集型 |
| 线程池 | `concurrent.futures` | CPU 密集型 |
| 进程池 | `multiprocessing` | 多模型并行 |
| 请求队列 | Redis/RabbitMQ | 削峰填谷 |
| 推理锁 | `threading.Lock()` | GPU 资源竞争 |

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 线程池处理 CPU 密集型任务
executor = ThreadPoolExecutor(max_workers=4)

async def process_request(data):
    # 异步包装同步代码
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, sync_inference, data)
    return result
```

#### 1.6.2 临时文件管理

**为什么：** 防止磁盘空间被临时文件耗尽。

```python
import tempfile
import threading
import time

class TempFileManager:
    """临时文件管理器 - 延迟清理"""

    def __init__(self, cleanup_delay: int = 300):
        self.cleanup_delay = cleanup_delay
        self.pending_files = {}
        self.lock = threading.Lock()

    def create_temp_file(self, suffix: str = ".tmp"):
        """创建临时文件，返回路径和清理函数"""
        import uuid
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{suffix}")

        def delayed_cleanup():
            time.sleep(self.cleanup_delay)
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

        return temp_path, delayed_cleanup

    def schedule_cleanup(self, temp_path: str):
        """调度延迟清理"""
        _, cleanup_fn = self.create_temp_file()
        threading.Thread(target=cleanup_fn, daemon=True).start()
```

#### 1.6.3 大文件流式处理

**为什么：** 避免大文件加载到内存导致 OOM。

| 场景 | 处理方式 | 阈值建议 |
|-----|---------|---------|
| 文件上传 | 流式写入临时文件 | > 50MB |
| URL 下载 | 分块下载到临时文件 | > 50MB |
| Base64 解码 | 大文件写入磁盘 | > 50MB |
| 二进制流 | 实时传输 | 直接读取 bytes |

**流式处理大文件：**

```python
import httpx

async def download_audio(url: str, temp_path: str):
    """流式下载大文件"""
    async with httpx.AsyncClient() as client:
        async with client.stream('GET', url, timeout=300) as response:
            with open(temp_path, 'wb') as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
```

**环境变量配置：**

```bash
# 允许部署时调整大文件阈值
LARGE_FILE_THRESHOLD_MB=50        # 大文件阈值
TEMP_FILE_CLEANUP_DELAY=300       # 临时文件清理延迟（秒）
TEMP_FILE_DIR=/tmp/app            # 临时文件目录
```

### 1.7 镜像构建与推送

#### 1.7.1 跨平台构建

**为什么：** 本地开发环境（ARM）与生产环境（AMD64）可能不同。

```bash
# 创建 buildx builder
docker buildx create --name mybuilder --driver docker-container --use

# 构建 linux/amd64 镜像
docker buildx build \
    --platform linux/amd64 \
    --tag hub.6scloud.com/namespace/image:tag \
    --push \
    .
```

#### 1.7.2 镜像标签规范

| 标签格式 | 示例 | 用途 |
|---------|------|------|
| 时间戳+Git标签 | `20250129-v1.2.3` | 生产发布 |
| Git commit | `a1b2c3d` | 开发测试 |
| 语义化版本 | `v1.2.3` | 稳定版本 |

**避免使用 `latest` 标签**，原因：
- 不可追溯，无法回滚
- 缓存问题，可能拉取旧版本
- 不利于版本管理

#### 1.7.3 构建记录管理

**为什么：** 跟踪所有镜像构建历史，便于问题追溯和版本管理。

**最佳实践：**
1. **自动记录构建信息**：每次构建自动记录到 `docker_build_record.md`
2. **记录关键信息**：构建日期、Git 提交、镜像地址、镜像大小、Bug 修复说明
3. **版本可追溯**：通过 Git 提交关联代码版本

**构建记录示例：**

```markdown
## 20250128-8cc68fb

- **构建日期**: 2026-01-28T11:52:08+08:00
- **Git 提交**: `8cc68fb04a2062187b947baec10d3dd5baf3d57f`
- **镜像地址**: `hub.6scloud.com/namespace/image:20250128-8cc68fb`
- **镜像大小**: 6.08GB
- **平台**: linux/amd64
- **Bug 修复**:
  - 修复 BytesIO 指针重置问题 (P0 Bug)
```

**构建脚本示例：**

```bash
# build_and_push.sh
record_build() {
    local version_tag=$1
    local full_image=$2
    local build_date=$(date -Iseconds)
    local git_commit=$(git rev-parse HEAD)

    cat >> docker_build_record.md <<EOF
## ${version_tag}
- **构建日期**: ${build_date}
- **Git 提交**: \`${git_commit}\`
- **镜像地址**: \`${full_image}\`
EOF
}
```

### 1.8 部署配置检查清单

#### 1.8.1 通用检查项

```markdown
## 部署前检查清单（通用）

### API 规范
- [ ] 实现了 `/health` 存活检查接口
- [ ] 实现了 `/ready` 就绪检查接口
- [ ] **【关键】`/ready` 执行实际业务测试，不只是形式化检查**
- [ ] 兼容行业标准 API 协议（OpenAI/Anthropic）
- [ ] 接口返回格式符合规范

### 健康检查（关键）
- [ ] `/health` 简单快速，只检查进程状态
- [ ] `/ready` 检查系统资源（内存、GPU）
- [ ] `/ready` 检查模型加载状态
- [ ] **【关键】`/ready` 执行实际推理测试，验证模型能正常输出**
- [ ] 推理测试失败返回 503，触发 K8s 摘除流量
- [ ] readinessProbe 配置合理的超时时间（推理可能需要较长时间）

### K8s 优雅退出（关键）
- [ ] 实现 SIGTERM 信号处理器
- [ ] 跟踪活跃请求数量
- [ ] 关闭期间拒绝新请求（返回 503）
- [ ] 配置 `terminationGracePeriodSeconds`
- [ ] 配置 `preStop` 钩子（可选）
- [ ] 测试优雅退出行为

### 日志与监控
- [ ] 日志包含 Trace ID
- [ ] 记录请求处理时间
- [ ] 记录资源使用情况
- [ ] 错误信息详细可追踪

### 容器化
- [ ] 使用多阶段构建
- [ ] 镜像体积合理（< 10GB）
- [ ] 使用非 root 用户运行
- [ ] 包含 HEALTHCHECK 指令
- [ ] 暴露正确的服务端口

### 构建与推送
- [ ] 使用 `docker buildx --platform linux/amd64` 构建
- [ ] 镜像标签格式为 `YYYYMMDD-git_tag`
- [ ] 不使用 `latest` 标签
- [ ] 建立构建记录文件（`docker_build_record.md`）
- [ ] 记录包含：日期、Git 提交、镜像地址、Bug 修复说明

### 部署配置
- [ ] 环境变量配置完整
- [ ] 所有配置项可通过环境变量调整
- [ ] 资源限制配置合理
- [ ] 健康检查参数合理
- [ ] 卷挂载配置正确
- [ ] 大文件阈值可通过环境变量配置
- [ ] 临时文件清理延迟可通过环境变量配置
```

---

## 第二部分：语音/音频领域经验

### 2.1 音频格式支持

#### 2.1.1 格式覆盖

**为什么：** 用户上传的音频来源多样，格式各异。

| 格式 | 使用场景 | 支持方式 |
|-----|---------|---------|
| MP3 | 网络音频、音乐 | ffmpeg 解码 |
| WAV | 专业录音、无损 | 直接读取 |
| PCM | 实时流、嵌入式 | 原始数据 |
| M4A/AAC | 苹果设备录音 | ffmpeg 解码 |
| FLAC | 无损压缩 | ffmpeg 解码 |
| OGG | 开源音频 | ffmpeg 解码 |
| MP4/WEBM | 视频提取音频 | ffmpeg 提取 |

**实现建议：**

```python
import torchaudio

# 使用 torchaudio 加载多种格式
# 自动处理采样率转换和通道转换
def load_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)

    # 重采样
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # 转换为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, target_sr
```

#### 2.1.2 音频预处理

| 预处理步骤 | 原因 | 实现 |
|-----------|------|------|
| 重采样 | 模型需要固定采样率 | `torchaudio.transforms.Resample` |
| 单声道 | 减少计算量 | `waveform.mean(dim=0)` |
| 归一化 | 统一音量 | `waveform / max(abs(waveform))` |
| 静音切除 | 减少无效计算 | VAD 检测 |

### 2.2 输入方式支持

#### 2.2.1 多种输入源

| 输入方式 | 场景 | 处理方式 |
|---------|------|---------|
| 文件上传 | 网页/APP 上传 | `UploadFile` 读取 |
| URL | 外部存储 | `httpx` 异步下载 |
| Base64 | 嵌入式传输 | `base64.b64decode` |
| 二进制流 | 实时传输 | 直接读取 bytes |

**流式处理大文件：**

```python
import httpx

async def download_audio(url: str, temp_path: str):
    """流式下载大文件"""
    async with httpx.AsyncClient() as client:
        async with client.stream('GET', url, timeout=300) as response:
            with open(temp_path, 'wb') as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
```

#### 2.2.2 音频元信息提取

```python
from dataclasses import dataclass

@dataclass
class AudioMetadata:
    extension: str = ""
    sample_rate: int = 0
    duration_seconds: float = 0.0
    file_size: int = 0
    num_channels: int = 0

def extract_metadata(file_path):
    """提取音频元信息"""
    info = torchaudio.info(file_path)
    return AudioMetadata(
        extension=file_path.split('.')[-1],
        sample_rate=info.sample_rate,
        duration_seconds=info.num_frames / info.sample_rate,
        file_size=os.path.getsize(file_path),
        num_channels=info.num_channels
    )
```

### 2.3 语音模型特殊需求

#### 2.3.1 语言检测

| 策略 | 实现 | 适用场景 |
|-----|------|---------|
| 自动检测 | 模型自动识别 | 未知语言 |
| 指定语言 | 用户传入 lang 参数 | 已知语言，提高准确率 |
| 多语言混合 | 分段检测 | 多语言音频 |

```python
# 语言参数示例
class Language(str, Enum):
    auto = "auto"
    zh = "zh"      # 中文
    en = "en"      # 英文
    ja = "ja"      # 日语
    ko = "ko"      # 韩语
    yue = "yue"    # 粤语
```

#### 2.3.2 时间戳输出

**为什么：** 语音转写需要与原始音频对齐。

```python
{
    "text": "这是一段语音",
    "segments": [
        {
            "text": "这是",
            "start": 0.0,
            "end": 1.5
        },
        {
            "text": "一段语音",
            "start": 1.5,
            "end": 3.0
        }
    ]
}
```

#### 2.3.3 VAD（语音活动检测）

**为什么：** 减少静音部分的计算，提高处理速度。

```python
# 使用 Silero VAD
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# 获取语音片段
speech_timestamps = get_speech_timestamps(waveform, model, sampling_rate=16000)
```

### 2.4 音频领域检查清单

```markdown
## 语音服务部署检查清单

### 音频格式
- [ ] 支持 MP3/WAV/FLAC/M4A 等常见格式
- [ ] 支持采样率转换
- [ ] 支持单/双声道处理
- [ ] 支持大文件流式处理

### 输入方式
- [ ] 支持文件上传
- [ ] 支持 URL 下载
- [ ] 支持 Base64 编码
- [ ] 支持流式传输

### 语音特性
- [ ] 支持语言选择/自动检测
- [ ] 支持时间戳输出（可选）
- [ ] 支持 VAD 静音切除（可选）
- [ ] 支持说话人分离（可选）

### 性能优化
- [ ] 音频预处理（重采样、归一化）
- [ ] 批量处理支持
- [ ] 长音频分段处理
- [ ] GPU 内存管理
```

---

## 第三部分：实施指南

### 3.1 快速开始

#### 步骤 1：创建基础 API 服务

```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

# 健康检查
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}

# 业务接口
@app.post("/v1/predict")
async def predict(file: UploadFile = File(...)):
    # 处理逻辑
    return {"result": "success"}
```

#### 步骤 2：编写 Dockerfile

```dockerfile
FROM python:3.10-slim AS builder
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install -r requirements.txt

FROM python:3.10-slim
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY . .
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=10s CMD curl -sf http://localhost:8000/health || exit 1
EXPOSE 8000
CMD ["python", "main.py"]
```

#### 步骤 3：构建并推送

```bash
# 构建
docker buildx build --platform linux/amd64 -t my-service:test .

# 测试
docker run -d -p 8000:8000 my-service:test
curl http://localhost:8000/health

# 推送
docker tag my-service:test hub.6scloud.com/ns/image:20250129-v1.0
docker push hub.6scloud.com/ns/image:20250129-v1.0
```

### 3.2 常见问题

#### 通用问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 镜像体积过大 | 包含模型/依赖 | 多阶段构建 + 启动时下载 |
| 启动慢 | 模型加载耗时 | 模型预热 + 就绪检查延迟 |
| 内存溢出 | GPU 内存不足 | 批量大小控制 + 队列限流 |
| 健康检查失败 | 端口未暴露 | 检查 Dockerfile EXPOSE |
| 流量打到未就绪实例 | readiness probe 形式化 | 添加实际推理测试到 `/ready` |
| 实例实际不可用但未摘除 | readiness probe 未检测推理能力 | 在 `/ready` 中执行真实推理 |
| 临时文件堆积 | 清理逻辑缺失 | 实现延迟清理机制 + 环境变量配置 |
| 配置无法调整 | 硬编码配置 | 所有配置项环境变量化 |
| **滚动更新时请求失败** | **未实现优雅退出** | **实现 SIGTERM 处理 + 等待活跃请求完成** |
| **Pod 关闭时数据丢失** | **强制终止进程** | **配置 terminationGracePeriodSeconds** |

#### 语音特有问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 格式不支持 | 缺少解码器 | 安装 ffmpeg |
| 采样率错误 | 未重采样 | 统一转换为 16kHz |
| 大文件超时 | 同步读取阻塞 | 流式处理 + 异步下载 |
| 多语言识别差 | 语言检测错误 | 允许用户指定语言 |

---

## 附录：参考资源

- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [Anthropic API 文档](https://docs.anthropic.com/)
- [Docker Buildx 文档](https://docs.docker.com/buildx/)
- [Kubernetes 健康检查](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [Kubernetes 优雅退出](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination)
- [torchaudio 文档](https://pytorch.org/audio/stable/index.html)
