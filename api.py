# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
import base64
import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing_extensions import Annotated
from typing import List, Optional, Union
from enum import Enum
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO

TARGET_FS = 16000


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


model_dir = "iic/SenseVoiceSmall"
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"),
)

regex = r"<\|.*\|>"

app = FastAPI()


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
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")] = None,
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
):
    result = await audio_to_text(files, lang)
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


async def audio_to_text(files: list, lang: str = "auto"):
    """通用音频转文字逻辑"""
    audios = []
    for f in files:
        file_io = await load_audio_input(f)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)

        # transform to target sample
        if audio_fs != TARGET_FS:
            resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
            data_or_path_or_list = resampler(data_or_path_or_list)

        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)

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


class SiliconFlowResponse(BaseModel):
    text: str


@app.post("/v1/audio/transcriptions", response_model=SiliconFlowResponse)
@app.post("/audio/transcriptions", response_model=SiliconFlowResponse)
async def siliconflow_transcribe(
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
    if language:
        lang = language
    else:
        lang = "auto"

    result = await audio_to_text([file], lang)
    return SiliconFlowResponse(text=result[0]["text"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=50000)