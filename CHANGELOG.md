# SenseVoice API 服务变更日志

## [v1.0.0] - 2024-01-23

### 需求实现

#### 1. HTTP 网络请求音频转写

**文件**: `openai_api.py`

**实现要点**:
- 支持三种输入方式：
  - `file`: multipart/form-data 文件上传
  - `url`: 音频文件 URL (支持 HTTP/HTTPS，带认证的 URL)
  - `base64_data`: Base64 编码音频数据 (支持 data: URL 格式)

**支持的格式**:
- 音频: mp3, wav, pcm, m4a, flac, aac
- 视频: mp4 (ffmpeg 自动提取音频)

**API 端点**:
```
POST /v1/audio/transcriptions     # OpenAI 兼容格式
POST /api/recognition             # 内部 API 格式
GET  /health                      # 健康检查
GET  /metrics                     # Prometheus 指标
```

#### 2. OpenAI Transcriptions API 兼容

**实现要点**:
- 响应格式兼容 OpenAI API 规范
- 支持参数: `language`, `response_format`, `temperature`
- 响应字段: `text`, `language`, `duration`

**调用示例**:
```bash
# 文件上传
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file=@audio.mp3

# URL 方式
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F url="https://example.com/audio.wav"

# Base64 方式
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F base64_data="UklGRi..."
```

#### 3. GPU 并发处理 (30 并发)

**实现要点**:
- 使用 `ThreadPoolExecutor(max_workers=30)` 线程池
- 异步请求处理 (`async def` + `run_in_executor`)
- 单进程模式 + 线程池并发 (避免多进程模型重复加载)
- 支持容器水平扩展 (docker-compose replicas)

**配置**:
```python
# openai_api.py
_executor = ThreadPoolExecutor(max_workers=30)
```

#### 4. 容器化部署与 Prometheus

**Dockerfile 实现**:
- 两阶段构建 (builder + runtime)
- CUDA 12.1 + cuDNN 8
- 健康检查 (`/health` 端点)
- 自动模型缓存持久化

**Prometheus 指标**:
```
sensevoice_requests_total{status, input_type}
sensevoice_request_duration_seconds{input_type}
sensevoice_gpu_memory_bytes
sensevoice_inflight_requests
sensevoice_concurrent_requests
sensevoice_model_load_seconds
```

#### 5. 临时文件自动清理

**实现要点**:
- 每个请求使用独立临时目录 (`tempfile.mkdtemp`)
- 请求完成后自动清理 (`shutil.rmtree`)
- 异常时 finally 块确保清理

```python
temp_dir = tempfile.mkdtemp(prefix=f"sensevoice_{os.getpid()}_")
try:
    # 处理...
finally:
    if temp_dir:
        cleanup_temp_dir(temp_dir)
```

#### 6. 日志记录

**实现要点**:
- 文件 + 控制台双输出
- 请求 ID 追踪 (`uuid.uuid4()[:8]`)
- 完整错误堆栈 (`traceback`)
- 包含任务信息、耗时、状态码

---

### 新增文件

| 文件 | 说明 |
|------|------|
| `cog.yaml` | Cog 模型配置 |
| `predict.py` | Cog 预测入口 + 模型单例 |
| `openai_api.py` | OpenAI 兼容 API 服务 |
| `Dockerfile` | GPU 容器镜像 |
| `docker-compose.yaml` | 容器编排配置 (更新) |

---

### 修复的问题

| 问题 | 修复 |
|------|------|
| 模型重复加载 (每次请求创建新实例) | 使用全局单例 (`model` + 线程锁) |
| `--workers > 1` 多进程显存溢出 | 默认 `workers=1`，添加警告日志 |
| Prometheus label 为 None | 添加默认值处理 (`input_type or "unknown"`) |
| 临时目录清理不彻底 | 使用 `shutil.rmtree` 递归删除 |
| 请求日志缺失 | 添加请求日志中间件 |

---

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SENSEVOICE_DEVICE` | cuda | 设备选择 (cuda/cpu) |
| `MODELSCOPE_CACHE` | /app/models | 模型缓存目录 |
| `PRELOAD_MODEL` | 1 | 启动时预加载模型 |
| `API_PORT` | 8000 | API 服务端口 |
| `OMP_NUM_THREADS` | 4 | OpenMP 线程数 |
| `CUDA_MODULE_LOADING` | LAZY | CUDA 延迟加载 |
| `TEMP_DIR` | /tmp/sensevoice | 临时文件目录 |
| `LOG_DIR` | /app/logs | 日志目录 |

---

### 部署命令

```bash
# Docker 构建
docker build -t sensevoice-api .

# Docker 运行
docker run -d \
  --gpus all \
  --name sensevoice \
  -p 8000:8000 \
  -v sensevoice-models:/models \
  sensevoice-api

# Docker Compose
docker-compose up -d

# 扩展并发 (Kubernetes)
kubectl scale --replicas=3 deployment/sensevoice
```

---

### 测试命令

```bash
# 健康检查
curl http://localhost:8000/health

# Prometheus 指标
curl http://localhost:8000/metrics

# 转写测试 (文件)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@test.wav

# 转写测试 (URL)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F url="https://example.com/audio.mp3"

# 转写测试 (Base64)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F base64_data="UklGRi4AAABXQVZFZm10..."
```