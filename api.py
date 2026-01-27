# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
import base64
import httpx
import psutil
import time
import torch
import logging
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing_extensions import Annotated
from typing import List, Optional, Union
from enum import Enum
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
from dataclasses import dataclass

TARGET_FS = 16000

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
try:
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"),
    )
    _model_loaded = True
except Exception as e:
    _model_load_error = str(e)
    _model_loaded = False

regex = r"<\|.*\|>"

app = FastAPI()


_model_load_error = None
_last_request_time = 0  # 上次成功请求的时间戳
AUDIO_TEST_DIR = "test_audios"
AUDIO_TEST_COOLDOWN_SECONDS = 30  # 30 秒内有成功请求则跳过推理测试


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
            use_itn=False,
            batch_size_s=60,
        )

        return {"text": res[0].get("text", "")}, None
    except Exception as e:
        return None, str(e)


@app.get("/health")
async def health():
    """存活健康检查 (Liveness Probe)"""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """就绪健康检查 (Readiness Probe) - 检查模型和系统状态"""
    global _last_request_time
    current_time = time.time()
    errors = []

    # 1. 检查系统内存
    mem = psutil.virtual_memory()
    if mem.available < 512 * 1024 * 1024:  # < 512MB
        errors.append(f"insufficient_memory: available={mem.available/1024/1024:.0f}MB")
    if mem.percent > 90:
        errors.append(f"high_memory_usage: {mem.percent}%")

    # 2. 检查 GPU 状态
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated(0) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(0) / 1024 / 1024
        if allocated / gpu_mem.total_memory * 100 > 85:
            errors.append(f"high_gpu_memory: {allocated:.0f}MB/{gpu_mem.total_memory/1024/1024:.0f}MB ({allocated/gpu_mem.total_memory*100:.0f}%)")

    # 3. 检查模型加载状态
    if not _model_loaded or _model_load_error:
        errors.append(f"model_not_ready: {_model_load_error or 'unknown'}")

    # 4. 推理测试 (30 秒内无成功请求时执行)
    inference_result = None
    do_inference_test = True

    if _last_request_time > 0 and (current_time - _last_request_time) < AUDIO_TEST_COOLDOWN_SECONDS:
        do_inference_test = False  # 30 秒内有成功请求，跳过推理测试

    if do_inference_test:
        inference_result, inf_error = _perform_inference_test()
        if inf_error:
            errors.append(f"inference_failed: {inf_error}")

    if errors:
        return {"status": "not_ready", "errors": errors, "model": model_dir}, 503

    response = {
        "status": "ready",
        "model": model_dir,
        "memory": {
            "available_mb": round(mem.available / 1024 / 1024, 1),
            "usage_percent": mem.percent,
        },
    }

    # 如果进行了推理测试，添加推理结果信息
    if inference_result:
        response["inference_test"] = {
            "performed": True,
            "text_preview": inference_result["text"][:100] if inference_result["text"] else "",
        }
    else:
        response["inference_test"] = {
            "performed": False,
            "reason": "recent_request",
        }

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
    return {"result": result}


async def load_audio_input(file) -> BytesIO:
    """加载音频文件支持三种传入方式: file上传/base64编码/URL"""
    # 1. 直接文件上传 (UploadFile)
    if hasattr(file, 'read'):
        return BytesIO(await file.read())

    # 2. Base64 编码 (data:audio/xxx;base64,xxx 或纯 base64)
    if isinstance(file, str):
        if file.startswith('data:'):
            b64_data = file.split(',', 1)[1]
            audio_bytes = base64.b64decode(b64_data)
            return BytesIO(audio_bytes)

        if re.match(r'^[A-Za-z0-9+/=]+$', file) and len(file) > 100:
            audio_bytes = base64.b64decode(file)
            return BytesIO(audio_bytes)

        # 3. URL 方式
        if file.startswith('http://') or file.startswith('https://'):
            async with httpx.AsyncClient() as client:
                response = await client.get(file, timeout=300.0)
                response.raise_for_status()
                return BytesIO(response.content)

    raise ValueError(f"无法识别的音频文件格式")


async def audio_to_text(files: list, lang: str = "auto", request: Request = None):
    """通用音频转文字逻辑"""
    endpoint = request.url.path if request else "unknown"

    try:
        audios = []
        audio_infos = []
        for f in files:
            file_io = await load_audio_input(f)
            data_or_path_or_list, audio_fs = torchaudio.load(file_io)

            # 提取并记录音频元信息
            audio_meta = extract_audio_metadata(f, file_io, audio_fs)
            audio_infos.append(format_audio_metadata(audio_meta))

            # transform to target sample
            if audio_fs != TARGET_FS:
                resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
                data_or_path_or_list = resampler(data_or_path_or_list)

            data_or_path_or_list = data_or_path_or_list.mean(0)
            audios.append(data_or_path_or_list)

        # 记录音频信息
        if request is not None:
            for i, info in enumerate(audio_infos):
                logger.info(f"[{endpoint}] Audio {i+1}: {info}")

        if lang == "":
            lang = "auto"

        res = model.generate(
            input=audios,
            language=lang,
            use_itn=False,
            batch_size_s=60,
        )
        result = []
        for r in res:
            text = r["text"]
            result.append({
                "raw_text": text,
                "clean_text": re.sub(regex, "", text, 0, re.MULTILINE),
                "text": rich_transcription_postprocess(text),
            })
        return result

    except Exception as e:
        logger.error(f"[{endpoint}] Request failed: {type(e).__name__}: {str(e)}")
        raise


class SiliconFlowResponse(BaseModel):
    text: str


@app.post("/v1/audio/transcriptions", response_model=SiliconFlowResponse)
@app.post("/audio/transcriptions", response_model=SiliconFlowResponse)
async def siliconflow_transcribe(
    request: Request,
    file: Union[UploadFile, str] = File(..., description="Audio file: file upload, base64 encode, or URL"),
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

        if language:
            lang = language
        else:
            lang = "auto"

        # 提取音频元信息
        file_io = await load_audio_input(file)
        audio_meta = extract_audio_metadata(file, file_io)
        logger.info(f"[{endpoint}] Audio metadata: {format_audio_metadata(audio_meta)}")

        result = await audio_to_text([file], lang, request)
<<<<<<< HEAD

        global _last_request_time
        _last_request_time = time.time()  # 更新最后请求时间
=======
>>>>>>> e5f969e (Let me verify `httpx` is properly imported:)
        return SiliconFlowResponse(text=result[0]["text"])

    except ValueError as ve:
        logger.warning(f"[{endpoint}] Invalid request: {str(ve)}")
        raise
    except httpx.HTTPStatusError as he:
        logger.error(f"[{endpoint}] HTTP error fetching remote file: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"[{endpoint}] Processing failed: {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=50000)