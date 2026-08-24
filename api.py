# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re, math
import sys
import signal
import base64
import httpx
import psutil
import time
import torch
import logging
import tempfile
import threading
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
from typing_extensions import Annotated
from typing import List, Optional, Union, Callable
from enum import Enum
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
from dataclasses import dataclass

TARGET_FS = 16000

# 大文件流式处理配置
LARGE_FILE_THRESHOLD_MB = float(os.getenv("LARGE_FILE_THRESHOLD_MB", "50"))  # 大文件阈值(MB)，默认50MB
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "500"))  # 最大文件大小限制(MB)
TEMP_FILE_CLEANUP_DELAY = int(os.getenv("TEMP_FILE_CLEANUP_DELAY", "300"))  # 临时文件清理延迟(秒)，默认5分钟
TEMP_FILE_DIR = os.getenv("TEMP_FILE_DIR", "")  # 临时文件目录，为空时使用系统临时目录

# 性能优化配置
ENABLE_MODEL_WARMUP = os.getenv("ENABLE_MODEL_WARMUP", "false").lower() == "true"  # 默认不开启模型预热
ENABLE_INFERENCE_LOCK = os.getenv("ENABLE_INFERENCE_LOCK", "false").lower() == "true"  # 默认不开启推理锁

# K8s 优雅退出配置
GRACEFUL_SHUTDOWN_TIMEOUT = int(os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT", "30"))  # 优雅退出超时（秒）
_is_shutting_down = False
_active_requests = 0
_shutdown_event = asyncio.Event()

# Prometheus metrics
if _PROMETHEUS_AVAILABLE:
    ASR_REQUESTS_TOTAL = Counter(
        "asr_requests_total", "Total ASR requests", ["endpoint", "status"]
    )
    ASR_REQUEST_DURATION = Histogram(
        "asr_request_duration_seconds", "ASR request duration", ["endpoint"]
    )
    ASR_INFERENCE_DURATION = Histogram(
        "asr_inference_duration_seconds", "ASR inference duration", ["backend"]
    )
    ASR_AUDIO_DURATION = Histogram(
        "asr_audio_duration_seconds", "Input audio duration in seconds"
    )
    ASR_GPU_MEMORY_BYTES = Gauge(
        "asr_gpu_memory_bytes", "GPU memory allocated", ["device"]
    )
    ASR_ACTIVE_REQUESTS = Gauge(
        "asr_active_requests", "Current active requests"
    )
    ASR_MODEL_LOADED = Gauge(
        "asr_model_loaded", "Model loaded status (1=loaded, 0=not)"
    )

# 临时文件管理器
class TempFileManager:
    """管理临时音频文件的创建和延迟清理"""

    def __init__(self, cleanup_delay: int = 300, temp_dir: str = None):
        self.cleanup_delay = cleanup_delay
        self.temp_dir = temp_dir if temp_dir else tempfile.gettempdir()
        self.pending_files: dict = {}  # {file_path: (mtime, file_size)}
        self.lock = threading.Lock()
        self._cleanup_thread = None
        self._running = False

    def create_temp_file(self, suffix: str = ".wav") -> tuple[str, Callable]:
        """创建临时文件，返回(文件路径, 清理函数)"""
        import uuid
        temp_file = os.path.join(self.temp_dir, f"sensevoice_{uuid.uuid4().hex}{suffix}")
        file_size = 0

        def cleanup():
            """延迟清理临时文件"""
            import time
            time.sleep(self.cleanup_delay)
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temp file: {temp_file} ({file_size} bytes)")
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

        # 记录待清理文件
        with self.lock:
            self.pending_files[temp_file] = None  # 仅记录路径

        return temp_file, cleanup

    def track_file_size(self, path: str, size: int):
        """跟踪文件大小"""
        with self.lock:
            if path in self.pending_files:
                self.pending_files[path] = size

    def cleanup_immediately(self, path: str):
        """立即清理临时文件"""
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Immediately cleaned up temp file: {path}")
        except OSError as e:
            logger.warning(f"Failed to cleanup temp file {path}: {e}")
        finally:
            with self.lock:
                self.pending_files.pop(path, None)

    def cleanup_all(self):
        """清理所有待清理的文件"""
        with self.lock:
            paths = list(self.pending_files.keys())
        for path in paths:
            self.cleanup_immediately(path)

# 全局临时文件管理器实例
temp_file_manager = TempFileManager(
    cleanup_delay=TEMP_FILE_CLEANUP_DELAY,
    temp_dir=TEMP_FILE_DIR if TEMP_FILE_DIR else None
)

# 配置日志 - 使用强制配置确保不被其他库覆盖
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Python 3.8+ 支持，强制重新配置
)
logger = logging.getLogger(__name__)
# 确保 logger 级别正确设置
logger.setLevel(logging.INFO)


@dataclass
class AudioMetadata:
    """音频文件元信息"""
    extension: str = ""
    encoding: str = ""
    sample_rate: int = 0
    duration_seconds: float = 0.0
    file_size: int = 0
    num_channels: int = 0
    num_tracks: int = 0


def log_request_info(request: Request, endpoint: str, extra_info: dict = None):
    """记录请求信息"""
    # 1. 记录请求的具体接口
    client_host = request.client.host if request.client else "unknown"
    headers = request.headers

    # 2. 读取并记录 trace id headers
    trace_ids = {
        "X-Trace-Id": headers.get("X-Trace-Id"),
        "x-siliconcloud-trace-id": headers.get("x-siliconcloud-trace-id"),
    }

    log_msg = f"[{endpoint}] Request from {client_host}"
    log_msg += f" | Trace-Ids: {trace_ids}"

    if extra_info:
        log_msg += f" | Extra: {extra_info}"

    logger.info(log_msg)


def extract_audio_metadata(file, file_io: BytesIO, audio_fs: int = None) -> AudioMetadata:
    """提取音频文件的元信息"""
    metadata = AudioMetadata()

    # 文件扩展名
    if hasattr(file, 'filename') and file.filename:
        filename = file.filename
        metadata.extension = filename.split('.')[-1].lower() if '.' in filename else ""
    elif isinstance(file, str):
        if file.startswith('data:'):
            mime_type = file.split(';')[0].split(':')[-1] if ';' in file else ""
            metadata.extension = mime_type.split('/')[-1] if '/' in mime_type else ""
        elif file.startswith('http://') or file.startswith('https://'):
            metadata.extension = file.split('?')[0].split('.')[-1].lower() if '.' in file.split('?')[0] else "unknown"
        else:
            metadata.extension = "base64"

    # 文件大小
    if hasattr(file, 'size'):
        metadata.file_size = file.size
    else:
        pos = file_io.tell()
        file_io.seek(0, 2)
        metadata.file_size = file_io.tell()
        file_io.seek(pos)

    # 音频流信息
    try:
        info = torchaudio.info(file_io)
        metadata.sample_rate = info.sample_rate
        metadata.num_channels = info.num_channels
        metadata.num_tracks = info.num_frames if hasattr(info, 'num_frames') else 1
        metadata.duration_seconds = metadata.num_tracks / info.sample_rate if info.sample_rate > 0 else 0
    except Exception:
        if audio_fs:
            metadata.sample_rate = audio_fs
            metadata.duration_seconds = 0

    # 编码信息
    if hasattr(file, 'filename') and file.filename:
        metadata.encoding = "file_upload"
    elif isinstance(file, str):
        if file.startswith('data:'):
            metadata.encoding = "base64"
        elif file.startswith('http://') or file.startswith('https://'):
            metadata.encoding = "url"
        else:
            metadata.encoding = "base64"

    return metadata


def format_file_size(size_bytes: int) -> str:
    """将字节转换为易读的单位"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def format_audio_metadata(metadata: AudioMetadata) -> dict:
    """格式化音频元信息用于日志"""
    return {
        "extension": metadata.extension,
        "encoding": metadata.encoding,
        "sample_rate": f"{metadata.sample_rate}Hz",
        "duration_seconds": round(metadata.duration_seconds, 2),
        "file_size": format_file_size(metadata.file_size),
        "num_channels": metadata.num_channels,
    }


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


model_dir = "iic/SenseVoiceSmall"

# 标点模型配置 - 仅当手工指定时才加载
_punc_model = os.getenv("SENSEVOICE_PUNC_MODEL", "")

# GPU 兼容性检测函数
def _check_gpu_compatibility():
    """检查 GPU 兼容性，返回 (是否兼容, 警告信息列表)"""
    warnings = []

    if not torch.cuda.is_available():
        return True, warnings  # CPU 模式，无需检查

    try:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        cuda_version = torch.version.cuda
        pytorch_version = torch.__version__

        logger.info(f"GPU: {device_name}")
        logger.info(f"Compute Capability: {capability}")
        logger.info(f"CUDA Version: {cuda_version}")
        logger.info(f"PyTorch Version: {pytorch_version}")

        # 解析 PyTorch 版本 (处理 "2.3.0+cu121" 格式)
        try:
            version_parts = pytorch_version.split('+')[0].split('.')
            pytorch_major = int(version_parts[0])
            pytorch_minor = int(version_parts[1])
        except (ValueError, IndexError):
            logger.warning(f"无法解析 PyTorch 版本: {pytorch_version}，跳过版本检查")
            pytorch_major, pytorch_minor = 2, 3  # 假设兼容

        # Blackwell 架构 (RTX 50xx) 检测 - Compute Capability 10.0+
        if capability[0] >= 10:
            warnings.append(f"检测到 Blackwell 架构 GPU ({device_name})")
            warnings.append(f"Compute Capability: {capability[0]}.{capability[1]}")

            # 检查 PyTorch 版本 (RTX 5090 需要 PyTorch 2.7.0+，因为 2.6.0 在 cu128 中不可用)
            if pytorch_major < 2 or (pytorch_major == 2 and pytorch_minor < 7):
                warnings.append(f"ERROR: RTX 5090 需要 PyTorch >= 2.7.0，当前版本: {pytorch_version}")
                warnings.append("请升级 PyTorch: pip install torch>=2.7.0 torchaudio --index-url https://download.pytorch.org/whl/cu128")
                return False, warnings

            # 检查 CUDA 版本 (处理 "12.8" 或 "12.8.0" 格式)
            if cuda_version:
                try:
                    cuda_major_minor = float('.'.join(cuda_version.split('.')[:2]))
                    if cuda_major_minor < 12.8:
                        warnings.append(f"ERROR: RTX 5090 需要 CUDA >= 12.8，当前版本: {cuda_version}")
                        return False, warnings
                except ValueError:
                    logger.warning(f"无法解析 CUDA 版本: {cuda_version}，跳过版本检查")

        # Hopper/Ada/Ampere/Turing 等架构 (正常支持)
        else:
            logger.info(f"GPU 架构正常支持 (Compute Capability {capability[0]}.{capability[1]})")

    except Exception as e:
        logger.warning(f"GPU 兼容性检查失败: {e}")

    return True, warnings

try:
    # 首先检查 GPU 兼容性
    _gpu_compatible, _gpu_warnings = _check_gpu_compatibility()
    if not _gpu_compatible:
        for warning in _gpu_warnings:
            logger.error(warning)
        raise RuntimeError(f"GPU 不兼容: {'; '.join(_gpu_warnings)}")

    model_kwargs = {
        "model": model_dir,
        "trust_remote_code": True,
        "remote_code": "./model.py",
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
        "device": os.getenv("SENSEVOICE_DEVICE", "cuda:0"),
    }

    # 仅当手工指定标点模型时才加载
    if _punc_model:
        model_kwargs["punc_model"] = _punc_model

    model = AutoModel(**model_kwargs)

    # 启动时进行真正的推理测试，确保模型可加载
    import torch
    import torchaudio
    test_waveform = torch.zeros(16000, dtype=torch.float32)  # 1秒静音
    _ = model.generate(
        input=test_waveform,
        language="auto",
        use_itn=True,
        batch_size_s=60,
    )

    # 可选：模型预热（通过环境变量开启）
    if ENABLE_MODEL_WARMUP:
        logger.info("开始模型预热...")
        warmup_start = time.time()

        # 预热 1: 基础推理
        _ = model.generate(
            input=test_waveform,
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )

        # 预热 2: 再次推理确保 CUDA 上下文完全初始化
        _ = model.generate(
            input=test_waveform,
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )

        warmup_duration = time.time() - warmup_start
        logger.info(f"模型预热完成，耗时: {warmup_duration:.3f}s")

        # 设置 CUDA 为性能模式（如果可用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True  # 启用 cudnn 自动优化
            logger.info(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    _model_loaded = True
    _model_load_error = None
except Exception as e:
    _model_load_error = str(e)
    _model_loaded = False
    logger.error(f"模型加载失败: {e}")

regex = r"<\|.*\|>"

_last_request_time = 0  # 上次成功请求的时间戳
_last_ready_inference_time = 0.0  # 上次就绪探针推理测试时间戳
AUDIO_TEST_DIR = "test_audios"
READY_INFERENCE_COOLDOWN = int(os.getenv("READY_INFERENCE_COOLDOWN", "60"))  # 推理测试冷却时间（秒）
READY_INFERENCE_IDLE_THRESHOLD = int(os.getenv("READY_INFERENCE_IDLE_THRESHOLD", "60"))  # 业务请求空闲阈值（秒）

# 可选：模型推理锁，防止并发请求导致 GPU 资源竞争（通过环境变量开启）
_model_inference_lock = threading.Lock() if ENABLE_INFERENCE_LOCK else None


def handle_sigterm(signum, frame):
    """处理 SIGTERM 信号（K8s 发送）- 优雅退出"""
    global _is_shutting_down
    logger.info(f"Received SIGTERM, starting graceful shutdown...")
    _is_shutting_down = True

    # 创建异步任务等待请求完成
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(wait_for_requests_complete())
        else:
            asyncio.run(wait_for_requests_complete())
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")
        sys.exit(1)


async def wait_for_requests_complete():
    """等待所有活跃请求完成"""
    global _active_requests

    logger.info(f"Waiting for {_active_requests} active requests to complete...")

    # 等待最多 GRACEFUL_SHUTDOWN_TIMEOUT 秒
    waited = 0
    while _active_requests > 0 and waited < GRACEFUL_SHUTDOWN_TIMEOUT:
        await asyncio.sleep(1)
        waited += 1
        if waited % 5 == 0:  # 每 5 秒记录一次
            logger.info(f"Still waiting for {_active_requests} active requests... ({waited}s)")

    if _active_requests > 0:
        logger.warning(f"Graceful shutdown timeout. {_active_requests} requests remaining.")
    else:
        logger.info("Graceful shutdown complete. All requests finished.")

    # 清理所有临时文件
    try:
        temp_file_manager.cleanup_all()
        logger.info("Temp files cleaned up.")
    except Exception as e:
        logger.warning(f"Temp file cleanup error: {e}")

    _shutdown_event.set()
    sys.exit(0)


# 注册信号处理器
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Service starting...")
    yield
    logger.info("Service shutting down...")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def graceful_shutdown_middleware(request: Request, call_next):
    """中间件：处理优雅关闭期间的请求"""
    global _active_requests, _is_shutting_down

    # 如果正在关闭，拒绝新请求
    if _is_shutting_down:
        return JSONResponse(
            status_code=503,
            content={"error": "Service is shutting down", "status": "unavailable"}
        )

    # 增加活跃请求计数
    _active_requests += 1
    if _PROMETHEUS_AVAILABLE:
        ASR_ACTIVE_REQUESTS.set(_active_requests)
    start = time.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        # 减少活跃请求计数
        _active_requests -= 1
        if _PROMETHEUS_AVAILABLE:
            ASR_ACTIVE_REQUESTS.set(_active_requests)
            duration = time.time() - start
            ASR_REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
            ASR_REQUESTS_TOTAL.labels(endpoint=request.url.path, status=str(status_code)).inc()


def _perform_inference_test():
    """执行实际的模型推理测试"""
    import glob

    # 查找测试音频文件
    audio_patterns = [f"{AUDIO_TEST_DIR}/*.wav", f"{AUDIO_TEST_DIR}/*.mp3", f"{AUDIO_TEST_DIR}/*.flac"]
    test_files = []
    for pattern in audio_patterns:
        test_files.extend(glob.glob(pattern))

    if not test_files:
        return None, "no_test_audio"

    try:
        test_file = test_files[0]
        waveform, audio_fs = torchaudio.load(test_file)

        # 重采样至 16kHz
        if audio_fs != TARGET_FS:
            resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
            waveform = resampler(waveform)

        waveform = waveform.mean(0)

        # 执行推理
        res = model.generate(
            input=[waveform],
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )

        return {"text": res[0].get("text", "")}, None
    except Exception as e:
        return None, str(e)


@app.get("/health")
async def health():
    """存活健康检查 (Liveness Probe)"""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    if not _PROMETHEUS_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "prometheus-client not installed"},
        )
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/ready")
async def ready():
    """就绪健康检查 (Readiness Probe) - 检查模型、资源，并执行实际推理测试（带冷却和空闲检测）"""
    global _last_ready_inference_time
    errors = []
    current_time = time.time()

    # Update Prometheus gauges
    if _PROMETHEUS_AVAILABLE:
        ASR_MODEL_LOADED.set(1 if _model_loaded else 0)

    # 1. 检查模型加载状态
    if not _model_loaded or _model_load_error:
        errors.append(f"model_not_ready: {_model_load_error or 'unknown'}")

    # 2. 检查系统内存
    try:
        mem = psutil.virtual_memory()
        if mem.available < 512 * 1024 * 1024:  # < 512MB
            errors.append(f"insufficient_memory: available={mem.available/1024/1024:.0f}MB")
        if mem.percent > 90:
            errors.append(f"high_memory_usage: {mem.percent}%")
    except Exception as e:
        errors.append(f"memory_check_failed: {str(e)}")

    # 3. 检查 GPU 状态
    if torch.cuda.is_available():
        try:
            gpu_mem = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0)
            total = gpu_mem.total_memory
            if _PROMETHEUS_AVAILABLE:
                ASR_GPU_MEMORY_BYTES.labels(device="0").set(allocated)
            if allocated / total > 0.95:
                errors.append(f"high_gpu_memory: {allocated/1024/1024:.0f}MB/{total/1024/1024:.0f}MB ({allocated/total*100:.0f}%)")
        except Exception as e:
            errors.append(f"gpu_check_failed: {str(e)}")

    # 4. 实际推理测试（带冷却和空闲检测）
    inference_test_performed = False
    inference_test_skipped = False
    should_test = (
        _last_ready_inference_time == 0
        or (current_time - _last_ready_inference_time) > READY_INFERENCE_COOLDOWN
    )

    if should_test and _model_loaded:
        # 如果近期有成功业务请求，跳过推理测试
        if _last_request_time > 0 and (current_time - _last_request_time) < READY_INFERENCE_IDLE_THRESHOLD:
            inference_test_skipped = True
            logger.debug("/ready: skipping inference test, recent business requests detected")
        else:
            inference_result, inf_error = _perform_inference_test()
            if inf_error:
                errors.append(f"inference_test_failed: {inf_error}")
            else:
                _last_ready_inference_time = current_time
                inference_test_performed = True
                logger.debug("/ready: inference test passed")

    if errors:
        logger.warning(f"/ready check failed: {errors}")
        return {"status": "not_ready", "errors": errors, "model": model_dir}, 503

    response = {
        "status": "ready",
        "model": model_dir,
    }

    if inference_test_performed:
        response["inference_test"] = {"performed": True}
    elif inference_test_skipped:
        response["inference_test"] = {"performed": False, "reason": "recent_requests"}
    else:
        response["inference_test"] = {"performed": False, "reason": "cooldown"}

    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.post("/api/v1/asr")
async def turn_audio_to_text(
    request: Request,
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")] = None,
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
):
    # 记录请求信息
    log_request_info(request, "/api/v1/asr")

    result = await audio_to_text(files, lang, request)
    global _last_request_time
    _last_request_time = time.time()  # 更新最后请求时间

    # 计算总音频时长（用于计费）
    total_duration = sum(r.get("duration_seconds", 0.0) for r in result)
    return {
        "result": result,
        "usage": {
            "type": "duration",
            "seconds": int(math.ceil(total_duration)),
        },
    }


async def download_url_with_retry(url: str, temp_path: str = None, timeout: float = 300.0) -> tuple[bytes, float]:
    """下载URL音频文件，支持指数退避重试（最多3次）"""
    last_exception = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, timeout=timeout)
                resp.raise_for_status()
                content = resp.content
                size_mb = len(content) / (1024 * 1024)
                if temp_path:
                    with open(temp_path, 'wb') as f:
                        f.write(content)
                return content, size_mb
        except Exception as e:
            last_exception = e
            wait_time = 2 ** attempt
            logger.warning(f"URL下载尝试 {attempt + 1} 失败: {url}: {e}，{wait_time}秒后重试...")
            await asyncio.sleep(wait_time)
    raise last_exception


async def get_file_size_mb(file) -> float:
    """获取上传文件或BytesIO的大小(MB)"""
    # UploadFile 类型
    if hasattr(file, 'size') and file.size:
        return file.size / (1024 * 1024)

    # BytesIO 类型
    if hasattr(file, 'getbuffer'):
        return file.getbuffer().nbytes / (1024 * 1024)

    return 0.0


async def load_audio_input_streaming(file, temp_path: str = None) -> tuple[str, str, float]:
    """
    流式加载大音频文件到临时文件

    Args:
        file: 上传的文件、base64编码或URL
        temp_path: 临时文件路径（流式写入模式）
    Returns:
        tuple: (temp_path, 临时文件路径, 文件大小MB)
    """
    is_large_file = temp_path is not None

    # 1. 直接文件上传 (UploadFile) - 流式写入临时文件
    if hasattr(file, 'read'):
        if is_large_file:
            # 大文件：流式写入临时文件
            with open(temp_path, 'wb') as f:
                chunk_size = 8192  # 8KB chunks
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            # 记录文件大小
            temp_file_manager.track_file_size(temp_path, os.path.getsize(temp_path))
            return temp_path, temp_path, await get_file_size_mb(file)
        else:
            # 小文件：直接读入内存
            return BytesIO(await file.read()), None, await get_file_size_mb(file)

    # 2. Base64 编码 - 始终加载到内存（因为是编码后的数据）
    if isinstance(file, str):
        if file.startswith('data:'):
            b64_data = file.split(',', 1)[1]
            audio_bytes = base64.b64decode(b64_data)
            size_mb = len(audio_bytes) / (1024 * 1024)

            if is_large_file and size_mb > LARGE_FILE_THRESHOLD_MB:
                # 大 base64 数据写入临时文件
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                temp_file_manager.track_file_size(temp_path, len(audio_bytes))
                return temp_path, temp_path, size_mb

            return BytesIO(audio_bytes), None, size_mb

        if re.match(r'^[A-Za-z0-9+/=]+$', file) and len(file) > 100:
            audio_bytes = base64.b64decode(file)
            size_mb = len(audio_bytes) / (1024 * 1024)

            if is_large_file and size_mb > LARGE_FILE_THRESHOLD_MB:
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                temp_file_manager.track_file_size(temp_path, len(audio_bytes))
                return temp_path, temp_path, size_mb

            return BytesIO(audio_bytes), None, size_mb

        # 3. URL 方式 - 带重试的下载
        if file.startswith('http://') or file.startswith('https://'):
            # 先检查文件大小
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    head_resp = await client.head(file, timeout=30.0)
                    content_length = head_resp.headers.get('content-length')
                    size_mb = float(content_length) / (1024 * 1024) if content_length else 0.0
                    if size_mb > MAX_FILE_SIZE_MB:
                        raise ValueError(f"URL文件过大: {size_mb:.2f}MB (最大{MAX_FILE_SIZE_MB}MB)")
                except httpx.HTTPError:
                    size_mb = 0.0

            # 使用重试下载
            content, size_mb = await download_url_with_retry(file, temp_path if is_large_file else None, timeout=300.0)
            if size_mb > MAX_FILE_SIZE_MB:
                raise ValueError(f"下载文件过大: {size_mb:.2f}MB (最大{MAX_FILE_SIZE_MB}MB)")

            if is_large_file and temp_path:
                temp_file_manager.track_file_size(temp_path, os.path.getsize(temp_path))
                return temp_path, temp_path, size_mb
            else:
                return BytesIO(content), None, size_mb

    raise ValueError(f"无法识别的音频文件格式")


def load_audio_with_torchaudio(source, sample_rate: int = None) -> tuple[torch.Tensor, int]:
    """
    从文件路径或BytesIO加载音频

    Args:
        source: 文件路径(str)或BytesIO对象
        sample_rate: 目标采样率
    Returns:
        tuple: (音频tensor, 原始采样率)
    """
    waveform, audio_fs = torchaudio.load(source)

    # 重采样至目标采样率
    if sample_rate and audio_fs != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=sample_rate)
        waveform = resampler(waveform)
        audio_fs = sample_rate

    # 转换为单声道
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    return waveform, audio_fs


async def load_audio_input(file) -> BytesIO:
    """加载音频文件到 BytesIO（同步/异步统一接口）"""
    # 1. 直接文件上传 (UploadFile) - 需要异步读取
    if hasattr(file, 'read'):
        content = await file.read()
        return BytesIO(content)

    # 2. Base64 编码 (data:audio/xxx;base64,xxx 或纯 base64) - 同步处理
    if isinstance(file, str):
        if file.startswith('data:'):
            b64_data = file.split(',', 1)[1]
            audio_bytes = base64.b64decode(b64_data)
            return BytesIO(audio_bytes)

        if re.match(r'^[A-Za-z0-9+/=]+$', file) and len(file) > 100:
            audio_bytes = base64.b64decode(file)
            return BytesIO(audio_bytes)

        # 3. URL 方式 - 异步下载
        if file.startswith('http://') or file.startswith('https://'):
            async with httpx.AsyncClient() as client:
                response = await client.get(file, timeout=300.0)
                response.raise_for_status()
                return BytesIO(response.content)

    raise ValueError(f"无法识别的音频文件格式")


async def audio_to_text(files: list, lang: str = "auto", request: Request = None, file_ios: list = None):
    """通用音频转文字逻辑，支持大文件流式处理

    Args:
        files: 原始文件对象列表 (UploadFile, str路径, 或 base64/URL)
        lang: 语言
        request: 请求对象
        file_ios: 已加载的 BytesIO 列表（可选，用于避免重复读取文件）
    """
    import time
    endpoint = request.url.path if request else "unknown"
    process_start = time.time()

    try:
        audios = []
        audio_infos = []
        temp_files_to_cleanup = []  # 记录需要清理的临时文件

        # 预处理：加载所有 BytesIO（如果传入）
        ios_pool = file_ios or [None] * len(files)
        file_io_iter = iter(ios_pool) if ios_pool else None

        for idx, f in enumerate(files):
            # 从池中获取预加载的 BytesIO（如果有），否则动态加载
            file_io = next(file_io_iter) if file_io_iter else None
            temp_path = None
            temp_file_manager_path = None

            # 如果没有预加载的 BytesIO，则需要加载
            if file_io is None:
                # 检测文件大小
                file_size_mb = await get_file_size_mb(f)
                # 确定是否使用流式处理（大文件写入临时文件）
                use_streaming = file_size_mb > LARGE_FILE_THRESHOLD_MB

                if use_streaming:
                    logger.info(f"[{endpoint}] Large file detected ({file_size_mb:.2f}MB > {LARGE_FILE_THRESHOLD_MB}MB), using streaming mode")
                    temp_path, temp_file_manager_path, size_mb = await load_audio_input_streaming(f, temp_file_manager.create_temp_file()[0])
                    temp_files_to_cleanup.append(temp_path)
                else:
                    file_io = await load_audio_input(f)

            try:
                # 加载音频
                if temp_path:
                    # 从临时文件加载
                    waveform, audio_fs = load_audio_with_torchaudio(temp_path)
                    # 清理临时文件（延迟清理）
                    cleanup_path = temp_path
                    if temp_file_manager_path:
                        # 启动延迟清理线程
                        _, cleanup_fn = temp_file_manager.create_temp_file()
                        threading.Thread(target=cleanup_fn, daemon=True).start()
                else:
                    # 从内存加载
                    waveform, audio_fs = load_audio_with_torchaudio(file_io)

                # 提取并记录音频元信息
                audio_meta = extract_audio_metadata(f, file_io if file_io else BytesIO(), audio_fs)
                audio_infos.append(format_audio_metadata(audio_meta))

                # 重采样至目标采样率
                if audio_fs != TARGET_FS:
                    resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
                    waveform = resampler(waveform)
                    audio_fs = TARGET_FS

                waveform = waveform.mean(0)  # 转换为单声道
                audios.append(waveform)

            finally:
                # 确保 BytesIO 被关闭（临时文件不需要关闭）
                if temp_path is None and file_io and hasattr(file_io, 'close'):
                    try:
                        file_io.close()
                    except Exception:
                        pass

        # 记录音频信息
        if request is not None:
            for i, info in enumerate(audio_infos):
                logger.info(f"[{endpoint}] Audio {i+1}: {info}")

        if lang == "":
            lang = "auto"

        # 检查模型是否已加载
        if not _model_loaded:
            raise RuntimeError("Model not loaded. Please check model initialization.")

        # 可选：使用锁序列化模型推理（通过环境变量 ENABLE_INFERENCE_LOCK 开启）
        inference_start = time.time()
        if _model_inference_lock:
            with _model_inference_lock:
                lock_wait_duration = time.time() - inference_start
                if lock_wait_duration > 0.1:  # 如果等待锁超过100ms，记录日志
                    logger.info(f"[{endpoint}] Waited {lock_wait_duration:.3f}s for model lock")

                res = model.generate(
                    input=audios,
                    language=lang,
                    use_itn=True,
                    batch_size_s=60,
                )
        else:
            res = model.generate(
                input=audios,
                language=lang,
                use_itn=True,
                batch_size_s=60,
            )
        inference_duration = time.time() - inference_start
        logger.info(f"[{endpoint}] Model inference completed in {inference_duration:.3f}s")

        # Record Prometheus metrics
        if _PROMETHEUS_AVAILABLE:
            ASR_INFERENCE_DURATION.labels(backend="funasr").observe(inference_duration)
            for info in audio_infos:
                dur = info.get("duration_seconds", 0)
                if dur and dur > 0:
                    ASR_AUDIO_DURATION.observe(dur)

        result = []
        for idx, r in enumerate(res):
            text = r["text"]
            info = audio_infos[idx] if idx < len(audio_infos) else {}
            result.append({
                "raw_text": text,
                "clean_text": re.sub(regex, "", text, 0, re.MULTILINE),
                "text": rich_transcription_postprocess(text),
                "duration_seconds": info.get("duration_seconds", 0.0),
            })

        total_duration = time.time() - process_start
        logger.info(f"[{endpoint}] audio_to_text total time: {total_duration:.3f}s (inference: {inference_duration:.3f}s)")
        return result

    except Exception as e:
        logger.error(f"[{endpoint}] Request failed: {type(e).__name__}: {str(e)}")
        raise


class SiliconFlowResponse(BaseModel):
    text: str
    language: Optional[str] = None
    usage: Optional[dict] = None


def _extract_language_from_sensevoice(raw_text: str) -> Optional[str]:
    """Extract language from SenseVoice raw output tags like <|zh|>, <|en|>."""
    match = re.search(r'<\|([a-z]{2,3})\|>', raw_text)
    if match:
        lang_code = match.group(1)
        lang_map = {
            "zh": "Chinese", "en": "English", "ja": "Japanese",
            "ko": "Korean", "yue": "Cantonese", "auto": "Auto",
        }
        return lang_map.get(lang_code, lang_code)
    return None


class TranscriptionRequest(BaseModel):
    """JSON body 请求格式，用于 base64 或 URL 提交"""
    file: str  # base64 编码或 URL
    model: str = "FunAudioLLM/SenseVoiceSmall"
    language: Optional[str] = None


@app.post("/v1/audio/transcriptions", response_model=SiliconFlowResponse)
@app.post("/audio/transcriptions", response_model=SiliconFlowResponse)
async def siliconflow_transcribe(
    request: Request,
    file: Union[UploadFile, str] = File(None, description="Audio file: file upload, base64 encode, or URL"),
    model: str = Form(default="FunAudioLLM/SenseVoiceSmall", description="Model name"),
    language: str = Form(default=None, description="Language (auto, zh, en, yue, ja, ko)"),
):
    """
    SiliconFlow 兼容接口 / OpenAI 格式音频转文字

    - 支持三种文件传入方式:
      1. formdata 文件上传 (file: File)
      2. Base64 编码 (file: "data:audio/mp3;base64,...")
      3. 音频 URL (file: "https://...")

    - 自动重采样至 16kHz
    - 返回可读文本（含标点）
    """
    import time
    request_start_time = time.time()

    # 获取请求路径
    endpoint = request.url.path
    client_host = request.client.host if request.client else "unknown"
    headers = request.headers

    logger.info(f"[{endpoint}] Request received from {client_host}")

    try:
        # 记录 trace id
        trace_ids = {
            "X-Trace-Id": headers.get("X-Trace-Id"),
            "x-siliconcloud-trace-id": headers.get("x-siliconcloud-trace-id"),
        }
        logger.info(f"[{endpoint}] Trace-Ids: {trace_ids}")

        # 检查是否是 JSON body 请求 (base64 或 URL)
        content_type = headers.get("content-type", "")
        if "application/json" in content_type:
            # JSON body 请求
            body = await request.json()
            file_input = body.get("file")
            language = body.get("language") or body.get("lang") or language
            if not file_input:
                raise ValueError("JSON body must contain 'file' field")
            if not isinstance(file_input, str):
                raise ValueError("'file' field must be a string (base64 or URL)")
        else:
            # multipart/form-data 请求
            file_input = file
            if file_input is None:
                raise ValueError("Missing file field in form-data")

        if language:
            lang = language
        else:
            lang = "auto"

        # 提取音频元信息
        file_io = await load_audio_input(file_input)
        audio_meta = extract_audio_metadata(file_input, file_io)
        # 重置指针到开头，因为 extract_audio_metadata 内部调用 torchaudio.info 移动了指针
        file_io.seek(0)
        logger.info(f"[{endpoint}] Audio metadata: {format_audio_metadata(audio_meta)}")

        # 传入已加载的 file_io，避免重复读取导致文件指针失效
        result = await audio_to_text([file_input], lang, request, file_ios=[file_io])

        global _last_request_time
        _last_request_time = time.time()  # 更新最后请求时间

        request_duration = time.time() - request_start_time
        logger.info(f"[{endpoint}] Request completed in {request_duration:.3f}s")

        # Extract language from raw output and build usage info
        detected_language = _extract_language_from_sensevoice(result[0].get("raw_text", ""))
        duration_seconds = audio_meta.duration_seconds
        usage_info = {
            "type": "duration",
            "seconds": int(math.ceil(duration_seconds)),
        }

        return SiliconFlowResponse(
            text=result[0]["text"],
            language=detected_language,
            usage=usage_info,
        )

    except ValueError as ve:
        logger.warning(f"[{endpoint}] Invalid request: {str(ve)}")
        raise
    except httpx.HTTPStatusError as he:
        logger.error(f"[{endpoint}] HTTP error fetching remote file: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"[{endpoint}] Processing failed: {type(e).__name__}: {str(e)}")
        raise


# K8s 优雅退出支持
_shutdown_event = threading.Event()
_shutdown_timeout = int(os.getenv("SHUTDOWN_TIMEOUT", "30"))  # 优雅退出等待超时(秒)


def _signal_handler(signum, frame):
    """处理 SIGTERM/SIGINT 信号，设置优雅退出标志"""
    signal_name = signal.Signals(signum).name
    logger.info(f"Received {signal_name}, initiating graceful shutdown...")
    _shutdown_event.set()


# 注册信号处理器 (仅在主线程中)
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    logger.info("Signal handlers registered for graceful shutdown")
except ValueError:
    # 非主线程时不注册信号处理器
    pass


if __name__ == "__main__":
    import uvicorn

    # 检查是否有通过环境变量设置端口（API_PORT 优先，兼容 PORT）
    port = int(os.getenv("API_PORT", os.getenv("PORT", "50000")))
    host = os.getenv("HOST", "0.0.0.0")

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=_shutdown_timeout,
    )
    server = uvicorn.Server(config)
    logger.info(f"Starting server on {host}:{port}, shutdown_timeout={_shutdown_timeout}s")
    server.run()