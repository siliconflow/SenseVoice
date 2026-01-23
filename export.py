import os
import torch
from funasr import AutoModel
from utils import export_utils
from utils.model_bin import SenseVoiceSmallONNX
from funasr.utils.postprocess_utils import rich_transcription_postprocess

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
model_path = os.path.dirname(model_wrapper.model_path)

rebuilt_model = model.export(type="onnx", quantize=False)
exported_model_file = os.path.join(model_path, "model.onnx")
print("Export model onnx to {}".format(exported_model_file))

export_utils.export(model=rebuilt_model, output_dir=model_path)
print("Export meta to {}".format(os.path.join(model_path, "config.yaml")))

model_bin = SenseVoiceSmallONNX(model_path)

try:
    from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
    tokenizer = SentencepiecesTokenizer(bpemodel=os.path.join(model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"))
except:
    tokenizer = None

text = ["<|woitn|><|NEUTRAL|><|zh|>你好世界"]
print("src_text: {}".format(text))
tokens = tokenizer.encode(text)
print("token: {}".format(tokens))

res = model_bin(wav_or_scp=tokenizer, language_list=[3], textnorm_list=[15])
print("infer res: {}".format(res))
print([rich_transcription_postprocess(i) for i in res])