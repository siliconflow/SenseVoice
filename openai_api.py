# SenseVoice OpenAI 兼容 API 服务
# 兼容 OpenAI Transcriptions API 格式

import os
import sys
import time
import uuid
import base64
import asyncio
import logging
import tempfile
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel
import uvicorn
from starlette.responses import Response

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ============== 配置 ==============
TEMP_DIR = "/tmp/sensevoice"
LOG_DIR = "/app/logs"

# 确保目录存在
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("/app/models", exist_ok=True)

# 配置日志
log_file = os.path.join(LOG_DIR, f"sensevoice_{os.getpid()}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("sensevoice-api")

# 配置环境变量
os.environ["SENSEVOICE_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============== Prometheus Metrics ==============
REQUEST_COUNTER = Counter(
    'sensevoice_requests_total',
    'Total number of requests',
    ['status', 'input_type']
)

REQUEST_LATENCY = Histogram(
    'sensevoice_request_duration_seconds',
    'Request duration in seconds',
    ['input_type'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

GPU_MEMORY_USAGE = Gauge(
    'sensevoice_gpu_memory_bytes',
    'GPU memory usage in bytes'
)

INFLIGHT_REQUESTS = Gauge(
    'sensevoice_inflight_requests',
    'Number of requests currently being processed'
)

MODEL_LOAD_TIME = Histogram(
    'sensevoice_model_load_seconds',
    'Model loading time in seconds',
    buckets=[5, 10, 20, 30, 60, 120, 300]
)

CONCURRENT_REQUESTS = Gauge(
    'sensevoice_concurrent_requests',
    'Current concurrent processing count'
)


# ============== 数据模型 ==============
class TranscriptionRequest(BaseModel):
    """OpenAI 兼容的转写请求"""
    file: Optional[str] = None
    model: str = "sensevoice"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = None
    temperature: Optional[float] = None
    timestamp_granularities: Optional[List[str]] = None

    class Config:
        extra = "allow"


class TranscriptionResponse(BaseModel):
    """转写响应"""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    chunks: Optional[List[Dict]] = None


# ============== 全局变量 ==============
model = None
model_load_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=30)  # 支持30并发
request_counter_lock = threading.Lock()


def get_model_name():
    """获取模型名称，可通过环境变量自定义"""
    return os.environ.get("MODEL_NAME", "iic/SenseVoiceSmall")


def preload_model():
    """预下载模型权重（不加载到内存，仅下载到缓存目录）"""
    model_name = get_model_name()
    logger.info(f"开始预下载模型: {model_name}")

    try:
        from modelscope import snapshot_download
        from funasr import AutoModel

        # 使用 modelscope 下载模型到缓存目录
        cache_dir = os.environ.get("MODELSCOPE_CACHE", "/app/models")
        os.makedirs(cache_dir, exist_ok=True)

        # 下载模型（不加载到内存）
        logger.info(f"从 ModelScope 下载模型到 {cache_dir} ...")
        model_path = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            allow_file_pattern=["*.bin", "*.safetensors", "*.pt", "*.py"],
        )
        logger.info(f"模型预下载完成: {model_path}")

        return True
    except Exception as e:
        logger.error(f"模型预下载失败: {e}")
        return False


def load_model():
    """加载模型（线程安全，单例模式）"""
    global model
    if model is not None:
        return

    with model_load_lock:
        if model is not None:
            return

        logger.info("=" * 50)
        logger.info("正在加载 SenseVoice 模型...")
        load_start = time.time()

        from funasr import AutoModel

        device = os.environ.get("SENSEVOICE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        # 检查是否只需要预下载
        if os.environ.get("ONLY_PRELOAD", "0") == "1":
            logger.info("ONLY_PRELOAD=1，仅执行模型预下载，不加载到内存")
            preload_model()
            logger.info("预下载完成，退出进程")
            sys.exit(0)

        model = AutoModel(
            model=get_model_name(),
            trust_remote_code=True,
            device=device,
        )

        load_time = time.time() - load_start
        MODEL_LOAD_TIME.observe(load_time)

        logger.info(f"SenseVoice 模型加载完成，耗时: {load_time:.2f}s")
        logger.info("=" * 50)

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            mem_total = props.total_memory / 1024**3
            logger.info(f"GPU: {props.name}")
            logger.info(f"GPU总显存: {mem_total:.2f} GB")
            logger.info(f"CUDA 版本: {torch.version.cuda}")


def cleanup_temp_file(file_path: str):
    """安全的临时文件清理"""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"已清理临时文件: {file_path}")
    except Exception as e:
        logger.warning(f"清理临时文件失败: {file_path}, 错误: {e}")


def cleanup_temp_dir(temp_dir: str):
    """安全的临时目录清理"""
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"已清理临时目录: {temp_dir}")
    except Exception as e:
        logger.warning(f"清理临时目录失败: {temp_dir}, 错误: {e}")


def download_file(url: str, temp_dir: str) -> str:
    """下载远程文件到临时目录"""
    import urllib.request
    import urllib.parse

    logger.info(f"开始下载远程文件: {url}")

    # 生成唯一文件名
    ext = Path(url).suffix or ".audio"
    if len(ext) > 10:  # 防止异常扩展名
        ext = ".audio"
    file_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(temp_dir, f"audio_{file_id}{ext}")

    try:
        # 处理带认证的URL
        parsed = urllib.parse.urlparse(url)
        headers = {}
        if parsed.username:
            auth = f"{parsed.username}:{parsed.password}"
            auth_base64 = base64.b64encode(auth.encode()).decode()
            headers = {"Authorization": f"Basic {auth_base64}"}

        request = urllib.request.Request(url, headers=headers, timeout=300)
        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            logger.info(f"文件大小: {total_size / 1024 / 1024:.2f} MB")

            downloaded = 0
            chunk_size = 8192
            with open(file_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

        logger.info(f"文件下载完成: {file_path} ({downloaded / 1024 / 1024:.2f} MB)")
        return file_path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {url}, 错误: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"下载文件失败: {str(e)}")


def decode_base64_audio(base64_data: str, temp_dir: str) -> str:
    """解码 base64 音频数据"""
    logger.info("开始解码 base64 音频数据")

    try:
        # 处理 data URL 格式 (data:audio/wav;base64,...)
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]

        # 验证 base64 格式
        if len(base64_data) < 100:
            raise HTTPException(status_code=400, detail="Base64 数据太短")

        audio_data = base64.b64decode(base64_data)
        if len(audio_data) < 100:
            raise HTTPException(status_code=400, detail="解码后的音频数据太小")

        file_id = str(uuid.uuid4())[:8]
        file_path = os.path.join(temp_dir, f"audio_{file_id}.wav")

        with open(file_path, 'wb') as f:
            f.write(audio_data)

        logger.info(f"Base64 音频解码完成: {file_path} ({len(audio_data) / 1024:.2f} KB)")
        return file_path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 解码失败: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Base64 解码失败: {str(e)}")


def update_gpu_metrics():
    """更新 GPU 指标"""
    if torch.cuda.is_available():
        try:
            mem_allocated = torch.cuda.memory_allocated()
            GPU_MEMORY_USAGE.set(mem_allocated)
        except Exception:
            pass


def process_transcription(
    audio_path: str,
    language: str = "auto",
    use_itn: bool = True,
) -> Dict[str, Any]:
    """执行转写处理"""
    start_time = time.time()

    if model is None:
        load_model()

    logger.info(f"开始转写: {os.path.basename(audio_path)}")
    logger.info(f"语言: {language}, ITN: {use_itn}")

    try:
        result = model.generate(
            input=audio_path,
            cache={},
            language=language,
            use_itn=1 if use_itn else 0,
            batch_size=1,
        )

        processing_time = time.time() - start_time
        update_gpu_metrics()

        if result and len(result) > 0:
            if isinstance(result[0], dict):
                text = result[0].get('text', '').strip()
            else:
                text = str(result[0]).strip()

            # 提取语言信息
            detected_lang = None
            for lang_tag in ['<|zh|>', '<|en|>', '<|yue|>', '<|ja|>', '<|ko|>']:
                if lang_tag in text:
                    detected_lang = lang_tag.replace('<|', '').replace('|>', '')
                    break

            logger.info(f"转写完成，耗时: {processing_time:.2f}s")
            logger.debug(f"转写结果: {text[:100]}...")

            return {
                "success": True,
                "text": text,
                "language": detected_lang or language,
                "duration": processing_time,
            }

        logger.warning("转写结果为空")
        return {
            "success": False,
            "error": "转写结果为空",
            "time": processing_time,
        }

    except Exception as e:
        logger.error(f"转写失败: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start_time,
        }


def process_audio_sync(
    audio_source: Any,
    input_type: str,
    language: str = "auto",
    use_itn: bool = True,
    response_format: str = "json",
) -> Dict[str, Any]:
    """同步处理音频转写"""
    temp_dir = None
    try:
        # 创建独立的临时目录
        temp_dir = tempfile.mkdtemp(prefix=f"sensevoice_{os.getpid()}_")

        audio_path = None

        # 处理不同类型的输入
        if input_type == "file":
            temp_file = os.path.join(temp_dir, "uploaded_audio")
            with open(temp_file, 'wb') as f:
                f.write(audio_source)
            audio_path = temp_file

        elif input_type == "url":
            audio_path = download_file(audio_source, temp_dir)

        elif input_type == "base64":
            audio_path = decode_base64_audio(audio_source, temp_dir)

        elif input_type == "path":
            audio_path = audio_source
        else:
            raise ValueError(f"不支持的输入类型: {input_type}")

        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail="音频文件处理失败")

        # 执行转写
        result = process_transcription(audio_path, language, use_itn)

        # 根据响应格式返回结果
        if response_format in ("json", "verbose_json"):
            return {
                "text": result.get("text", ""),
                "language": result.get("language"),
                "duration": result.get("duration"),
            }
        elif response_format == "text":
            return {"text": result.get("text", "")}
        else:
            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理音频失败: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 确保清理临时目录
        if temp_dir:
            cleanup_temp_dir(temp_dir)


async def process_audio(
    audio_source: Any,
    input_type: str,
    language: str = "auto",
    use_itn: bool = True,
    response_format: str = "json",
) -> Dict[str, Any]:
    """异步处理音频转写（使用线程池）"""
    # 确保 input_type 有效
    safe_input_type = input_type or "unknown"

    CONCURRENT_REQUESTS.inc()
    try:
        with REQUEST_LATENCY.labels(input_type=safe_input_type).time():
            result = await asyncio.get_event_loop().run_in_executor(
                _executor,
                process_audio_sync,
                audio_source,
                safe_input_type,
                language,
                use_itn,
                response_format,
            )

            # 线程安全计数器更新
            with request_counter_lock:
                REQUEST_COUNTER.labels(status="success", input_type=safe_input_type).inc()

            return result

    except Exception as e:
        with request_counter_lock:
            REQUEST_COUNTER.labels(status="error", input_type=safe_input_type).inc()
        raise


# ============== FastAPI 应用 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 可选：启动时预下载模型
    preload_on_start = os.environ.get("PRELOAD_MODEL_ON_START", "0") == "1"
    if preload_on_start:
        logger.info("PRELOAD_MODEL_ON_START=1，执行模型预下载...")
        preload_model()

    # 启动时加载模型（在主进程中）
    load_model()
    logger.info("SenseVoice API 服务启动完成，模型就绪")
    yield
    # 关闭时清理
    logger.info("SenseVoice API 服务关闭")


app = FastAPI(
    title="SenseVoice API",
    description="阿里 SenseVoice 语音识别 API - OpenAI 兼容格式",
    version="1.0.0",
    lifespan=lifespan,
)


# ============== 中间件 ==============
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] 完成，状态码: {response.status_code}, 耗时: {elapsed:.2f}s")
        return response
    except Exception as e:
        logger.error(f"[{request_id}] 异常: {str(e)}")
        raise


# ============== API 端点 ==============

@app.get("/health")
async def health_check():
    """健康检查"""
    gpu_status = "cuda" if torch.cuda.is_available() else "cpu"
    mem_info = ""
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        mem_info = f", GPU内存: {mem_allocated:.2f}/{mem_reserved:.2f} GB"

    return {
        "status": "ok",
        "model": "SenseVoiceSmall",
        "device": gpu_status,
        "gpu_info": mem_info,
        "worker_id": os.getpid(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics 端点"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "sensevoice",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "SenseVoice",
            }
        ]
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """获取模型信息"""
    if model_id != "sensevoice":
        raise HTTPException(status_code=404, detail="模型不存在")
    return {
        "id": "sensevoice",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "SenseVoice",
    }


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(None),
    model: str = Form("sensevoice"),
    language: Form(None) = Form(None),
    prompt: Form(None) = Form(None),
    response_format: Form("json") = Form(None),
    temperature: Form(None) = Form(None),
    timestamp_granularities: Form(None) = Form(None),
    url: str = Form(None),
    base64_data: str = Form(None),
):
    """
    OpenAI 兼容的音频转写 API

    支持多种输入方式:
    - file: 上传本地文件 (multipart/form-data)
    - url: 传入音频 URL (form data)
    - base64_data: 传入 base64 编码的音频数据 (form data)

    也可以直接发送 JSON Body，包含 'file' 字段
    """
    input_type = None
    audio_source = None

    # 尝试从 JSON Body 读取
    try:
        body = await request.json()
        json_file = body.get("file")
        json_url = body.get("url")
        json_base64 = body.get("base64_data")
    except Exception:
        json_file = json_url = json_base64 = None

    # 确定输入类型（优先级：file > url > base64 > JSON）
    if file and file.filename:
        content = await file.read()
        audio_source = content
        input_type = "file"
        logger.info(f"收到文件上传: {file.filename}, 大小: {len(content)} bytes")

    elif url:
        audio_source = url
        input_type = "url"
        logger.info(f"收到 URL 输入: {url}")

    elif base64_data:
        audio_source = base64_data
        input_type = "base64"
        logger.info(f"收到 Base64 输入")

    elif json_file and not (json_file.startswith("http") or json_file.startswith("data:")):
        audio_source = json_file
        input_type = "path"
        logger.info(f"收到文件路径输入: {json_file}")

    elif json_file and json_file.startswith(("http://", "https://")):
        audio_source = json_file
        input_type = "url"
        logger.info(f"收到 JSON URL 输入: {json_file}")

    elif json_base64:
        audio_source = json_base64
        input_type = "base64"
        logger.info(f"收到 JSON Base64 输入")

    elif json_file:
        audio_source = json_file
        input_type = "base64"
        logger.info(f"JSON file 按 base64 处理")

    else:
        raise HTTPException(
            status_code=400,
            detail="请提供 file、url、base64_data 参数或 JSON body"
        )

    # 构建处理参数
    lang = language if language else "auto"
    use_itn = True  # 默认启用 ITN
    fmt = response_format if response_format else "json"

    try:
        result = await process_audio(
            audio_source=audio_source,
            input_type=input_type,
            language=lang,
            use_itn=use_itn,
            response_format=fmt,
        )

        # OpenAI 格式响应
        if fmt in ("json", "verbose_json"):
            return {
                "text": result["text"],
                "language": result.get("language"),
                "duration": result.get("duration"),
            }
        else:
            return {"text": result["text"]}

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNTER.labels(status="error", input_type=input_type or "unknown").inc()
        logger.error(f"转写请求处理失败: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============== 内部 API ==============

@app.post("/api/recognition")
async def recognition_api(
    file: UploadFile = File(None),
    language: str = Form("auto"),
    use_itn: bool = Form(True),
    url: str = Form(None),
    base64_data: str = Form(None),
):
    """
    内部 API - 音频转写（兼容原 API 格式）
    """
    input_type = None
    audio_source = None

    if file and file.filename:
        content = await file.read()
        audio_source = content
        input_type = "file"
        logger.info(f"内部 API 收到文件: {file.filename}")

    elif url:
        audio_source = url
        input_type = "url"

    elif base64_data:
        audio_source = base64_data
        input_type = "base64"

    else:
        raise HTTPException(status_code=400, detail="请上传音频文件或提供 url/base64_data")

    result = await process_audio(
        audio_source=audio_source,
        input_type=input_type,
        language=language,
        use_itn=use_itn,
        response_format="json",
    )

    return result


# ============== 启动入口 ==============
def main():
    """启动 API 服务器"""
    import argparse

    parser = argparse.ArgumentParser(description="SenseVoice API Server")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数 (建议为1，多进程需独立部署)")
    parser.add_argument("--model-dir", type=str, default=None, help="模型缓存目录")
    args = parser.parse_args()

    # 设置模型缓存目录
    if args.model_dir:
        os.environ["MODELSCOPE_CACHE"] = args.model_dir

    # 模型预加载
    if os.environ.get("PRELOAD_MODEL", "1") == "1":
        load_model()

    logger.info(f"启动 SenseVoice API 服务器: {args.host}:{args.port}")
    logger.info(f"工作进程数: {args.workers} (建议: GPU服务器设为1，使用容器副本扩展)")

    # 使用单进程模式，避免多进程加载多个模型实例
    if args.workers > 1:
        logger.warning("警告: workers > 1 时每个进程会独立加载模型，可能导致显存溢出")

    uvicorn.run(
        "openai_api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
        log_config=None,  # 使用自定义日志配置
    )


if __name__ == "__main__":
    main()