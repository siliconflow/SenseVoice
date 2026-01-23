#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import torch
from funasr import AutoModel
from utils import export_utils
from utils.model_bin import SenseVoiceSmallONNX
from funasr.utils.postprocess_utils import rich_transcription_postprocess

quantize = False

model_dir = "iic/SenseVoiceSmall"
model_wrapper = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)
model = model_wrapper.model

rebuilt_model = model.export(type="onnx", quantize=False)
model_path = os.path.dirname(model_wrapper.model_path)

model_file = os.path.join(model_path, "model.onnx")
if quantize:
    model_file = os.path.join(model_path, "model_quant.onnx")

# export model
if not os.path.exists(model_file):
    with torch.no_grad():
        kwargs = {}
        export_dir = export_utils.export(model=rebuilt_model, **kwargs)
        print("Export model onnx to {}".format(model_file))

# export model init
model_bin = SenseVoiceSmallONNX(model_path)

# build tokenizer
try:
    from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
    tokenizer = SentencepiecesTokenizer(bpemodel=os.path.join(model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"))
except:
    tokenizer = None

# inference
wav_or_scp = "/Users/shixian/Downloads/asr_example_hotword.wav"
language_list = [0]
textnorm_list = [15]
res = model_bin(wav_or_scp, language_list, textnorm_list, tokenizer=tokenizer)
print([rich_transcription_postprocess(i) for i in res])