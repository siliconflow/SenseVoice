# SiliconFlow 平台部署最佳实践

## 概述

本 Skill 总结了 AI 服务部署到 SiliconFlow 平台的普适性最佳实践，适用于任何类型的 AI 服务（文本生成、语音处理、图像生成等）。

---

## 1. API 设计规范

### 1.1 兼容行业标准协议

**原则：** 兼容广泛使用的 API 规范可以大幅降低用户迁移成本。

| 场景 | 推荐协议 | 接口路径 |
|-----|---------|---------|
| 文本生成 | OpenAI Chat Completions | `/v1/chat/completions` |
| 文本转语音 | OpenAI Audio API | `/v1/audio/speech` |
| 语音转文本 | OpenAI Transcriptions | `/v1/audio/transcriptions` |
| 向量嵌入 | OpenAI Embeddings | `/v1/embeddings` |
| 多模态 | Anthropic Messages | `/v1/messages` |

**实施要点：**
- 保持请求/响应格式与标准协议一致
- 支持相同的认证方式（Bearer Token）
- 提供兼容的模型名称映射

### 1.2 健康检查接口设计（关键）

**核心原则：** Readiness probe 必须切实有效，不能只是形式化检查，要真实反映实例的服务响应能力。

**接口定义：**
- `/health`（Liveness）：简单快速，只检查进程是否活着
- `/ready`（Readiness）：必须执行实际的业务逻辑测试

```python
@app.get("/health")
async def health():
    """存活检查 - 简单快速"""
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    """就绪检查 - 必须真实反映服务响应能力"""
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

---

## 2. 容器化最佳实践

### 2.1 镜像体积控制

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

### 2.2 安全实践

| 实践 | 原因 | 实现 |
|-----|------|------|
| 非 root 用户 | 防止容器逃逸攻击 | `USER appuser` |
| 只读文件系统 | 防止运行时修改 | `readOnlyRootFilesystem: true` |
| 资源限制 | 防止资源耗尽 | `resources.limits` |
| 健康检查 | 自动恢复故障 | `HEALTHCHECK` 指令 |

---

## 3. K8s 优雅退出支持（关键）

### 3.1 为什么需要优雅退出

当 K8s 需要缩容、滚动更新或重启 Pod 时，会发送 SIGTERM 信号给容器。如果没有正确处理：
- 正在处理的请求会被强制中断
- 用户收到错误响应
- 数据可能不一致或丢失

### 3.2 优雅退出实现

```python
import signal
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

# 全局标志位，用于控制服务状态
_is_shutting_down = False
_active_requests = 0
_shutdown_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("Service starting...")
    yield
    # 关闭时
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
    max_wait = 30
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
        from fastapi.responses import JSONResponse
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

### 3.3 K8s 配置

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-service
spec:
  template:
    spec:
      # 关键：设置优雅退出超时时间
      terminationGracePeriodSeconds: 30

      containers:
      - name: api
        image: my-service:v1.0.0

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

### 3.4 优雅退出流程

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

### 3.5 检查清单

- [ ] 实现 SIGTERM 信号处理器
- [ ] 跟踪活跃请求数量
- [ ] 关闭期间拒绝新请求（返回 503）
- [ ] 配置 `terminationGracePeriodSeconds`
- [ ] 配置 `preStop` 钩子（可选，用于延迟关闭）
- [ ] 测试优雅退出行为

---

## 4. 日志与可观测性

### 4.1 结构化日志

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

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
    logger.info(f"[{endpoint}] {log_data}")
```

### 4.2 关键指标

| 指标 | 用途 | 示例 |
|-----|------|------|
| 请求处理时间 | 性能监控 | `total: 1.234s, inference: 0.523s` |
| 资源使用 | 容量规划 | `gpu_memory: 8.5GB/24GB` |
| 错误类型 | 问题定位 | `error: CUDA_OUT_OF_MEMORY` |

---

## 5. 配置管理

### 5.1 配置项环境变量化

**原则：** 所有可配置项都应通过环境变量暴露，便于部署时调整而不需要重新构建镜像。

| 配置类别 | 环境变量 | 默认值 | 说明 |
|---------|---------|-------|------|
| 模型配置 | `MODEL_NAME` | - | 模型名称 |
| | `DEVICE` | `cuda` | 运行设备 |
| | `ENABLE_WARMUP` | `false` | 是否开启预热 |
| 性能配置 | `LARGE_FILE_THRESHOLD_MB` | `50` | 大文件阈值 |
| | `TEMP_CLEANUP_DELAY` | `300` | 临时文件清理延迟 |
| | `OMP_NUM_THREADS` | `4` | OpenMP 线程数 |
| 优雅退出 | `GRACEFUL_SHUTDOWN_TIMEOUT` | `30` | 优雅退出超时（秒）|

```python
import os

# 所有配置项都有默认值，同时允许环境变量覆盖
LARGE_FILE_THRESHOLD_MB = float(os.getenv("LARGE_FILE_THRESHOLD_MB", "50"))
TEMP_CLEANUP_DELAY = int(os.getenv("TEMP_CLEANUP_DELAY", "300"))
ENABLE_WARMUP = os.getenv("ENABLE_WARMUP", "false").lower() == "true"
GRACEFUL_SHUTDOWN_TIMEOUT = int(os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT", "30"))
```

---

## 6. 模型加载策略

### 6.1 启动时下载 vs 镜像内置

| 方案 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 启动时下载 | 镜像小，模型可更新 | 首次启动慢 | 模型经常更新 |
| 镜像内置 | 启动快，无网络依赖 | 镜像大，模型固定 | 模型稳定，追求启动速度 |
| 共享卷挂载 | 启动快，镜像小 | 需要预置卷 | K8s 环境 |

### 6.2 模型预热

```python
def warmup_model(model):
    """模型预热 - 避免第一次请求冷启动"""
    import time
    start = time.time()

    dummy_input = torch.zeros(1, 3, 224, 224)
    _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Model warmup completed in {time.time() - start:.3f}s")
```

---

## 7. 并发与资源管理

### 7.1 并发控制策略

| 策略 | 实现 | 适用场景 |
|-----|------|---------|
| 异步处理 | `async/await` | I/O 密集型 |
| 线程池 | `concurrent.futures` | CPU 密集型 |
| 进程池 | `multiprocessing` | 多模型并行 |
| 请求队列 | Redis/RabbitMQ | 削峰填谷 |
| 推理锁 | `threading.Lock()` | GPU 资源竞争 |

### 7.2 临时文件管理

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
```

### 7.3 大文件流式处理

**原则：** 避免大文件加载到内存导致 OOM。

| 场景 | 处理方式 | 阈值建议 |
|-----|---------|---------|
| 文件上传 | 流式写入临时文件 | > 50MB |
| URL 下载 | 分块下载到临时文件 | > 50MB |
| Base64 解码 | 大文件写入磁盘 | > 50MB |

---

## 8. 镜像构建与推送

### 8.1 跨平台构建

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

### 8.2 镜像标签规范

| 标签格式 | 示例 | 用途 |
|---------|------|------|
| 时间戳+Git标签 | `20250129-v1.2.3` | 生产发布 |
| Git commit | `a1b2c3d` | 开发测试 |
| 语义化版本 | `v1.2.3` | 稳定版本 |

**避免使用 `latest` 标签**

### 8.3 构建记录管理

每次构建自动记录到 `docker_build_record.md`：

```markdown
## 20250128-8cc68fb

- **构建日期**: 2026-01-28T11:52:08+08:00
- **Git 提交**: `8cc68fb04a2062187b947baec10d3dd5baf3d57f`
- **镜像地址**: `hub.6scloud.com/namespace/image:20250128-8cc68fb`
- **镜像大小**: 6.08GB
- **平台**: linux/amd64
- **Bug 修复**:
  - 修复 XXX 问题 (P0 Bug)
```

---

## 9. 部署配置检查清单

```markdown
## 部署前检查清单

### API 规范
- [ ] 实现了 `/health` 存活检查接口
- [ ] 实现了 `/ready` 就绪检查接口
- [ ] `/ready` 执行实际业务测试，不只是形式化检查
- [ ] 兼容行业标准 API 协议（OpenAI/Anthropic）

### 健康检查（关键）
- [ ] `/health` 简单快速，只检查进程状态
- [ ] `/ready` 检查系统资源（内存、GPU）
- [ ] `/ready` 检查模型加载状态
- [ ] `/ready` 执行实际推理测试，验证模型能正常输出
- [ ] 推理测试失败返回 503，触发 K8s 摘除流量

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

### 部署配置
- [ ] 环境变量配置完整
- [ ] 所有配置项可通过环境变量调整
- [ ] 资源限制配置合理
- [ ] 健康检查参数合理
- [ ] 卷挂载配置正确
```

---

## 10. 常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 镜像体积过大 | 包含模型/依赖 | 多阶段构建 + 启动时下载 |
| 启动慢 | 模型加载耗时 | 模型预热 + 就绪检查延迟 |
| 内存溢出 | GPU 内存不足 | 批量大小控制 + 队列限流 |
| 流量打到未就绪实例 | readiness probe 形式化 | 添加实际推理测试到 `/ready` |
| 临时文件堆积 | 清理逻辑缺失 | 实现延迟清理机制 + 环境变量配置 |
| 配置无法调整 | 硬编码配置 | 所有配置项环境变量化 |
| **滚动更新时请求失败** | **未实现优雅退出** | **实现 SIGTERM 处理 + 等待活跃请求完成** |
| **Pod 关闭时数据丢失** | **强制终止进程** | **配置 terminationGracePeriodSeconds** |
