#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
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

# 支持通过环境变量配置标点模型
_use_punc = os.getenv("SENSEVOICE_USE_PUNC", "true").lower() == "true"
if _use_punc:
    model_kwargs["punc_model"] = os.getenv("SENSEVOICE_PUNC_MODEL", "iic/speech_punc_zh-cn-common-vocab2724-pytorch")

model = AutoModel(**model_kwargs)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)

# zh
res = model.generate(
    input=f"{model.model_path}/example/zh.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)

# yue
res = model.generate(
    input=f"{model.model_path}/example/yue.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)

# ja
res = model.generate(
    input=f"{model.model_path}/example/ja.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)


# ko
res = model.generate(
    input=f"{model.model_path}/example/ko.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
