#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/QwenAudio/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os

model_dir = "iic/SenseVoiceSmall"

model_kwargs = {
    "model": model_dir,
    "trust_remote_code": True,
    "remote_code": "./model.py",
    "vad_model": "fsmn-vad",
    "vad_kwargs": {"max_single_segment_time": 30000},
    "device": os.getenv("SENSEVOICE_DEVICE", "cuda:0"),
}

# 支持通过环境变量手工指定标点模型（默认使用 ITN）
_punc_model = os.getenv("SENSEVOICE_PUNC_MODEL", "")
if _punc_model:
    model_kwargs["punc_model"] = _punc_model

model = AutoModel(**model_kwargs)

# Process example audio files for different languages
example_files = ["en.mp3", "zh.mp3", "yue.mp3", "ja.mp3", "ko.mp3"]

for filename in example_files:
    res = model.generate(
        input=f"{model.model_path}/example/{filename}",
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    print(f"[{filename}] {text}")
