# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
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
    audios = []
    for file in files:
        file_io = BytesIO(await file.read())
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)

        # transform to target sample
        if audio_fs != TARGET_FS:
            resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
            data_or_path_or_list = resampler(data_or_path_or_list)

        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)

    if lang == "":
        lang = "auto"

    if not keys:
        key = [f.filename for f in files]
    else:
        key = keys.split(",")

    res = model.generate(
        input=audios,
        language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
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
    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=50000)
