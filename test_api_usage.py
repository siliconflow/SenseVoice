"""
单元测试：usage 字段取整逻辑（ceil 对齐 vLLM）+ 响应结构断言。

测试策略：
`api.py` 模块级即 `model = AutoModel(...)`（api.py:357），直接 import 会触发
FunASR 模型加载与 GPU 兼容性检查。因此 在 import 之前向 sys.modules 注入
fake 的 funasr / funasr.utils / torchaudio / torch / psutil / prometheus_client
模块，使 import api 不触发真实加载、也不依赖 GPU。
然后用 fastapi.testclient.TestClient 打两个端点，断言 usage.seconds 为
int(math.ceil(duration)) 的向上取整结果。

依赖：pytest、httpx（已含）、fastapi（已含）。
"""

import io
import math
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# 1. 注入 fake 模块，必须在 import api 之前
# ---------------------------------------------------------------------------

# fake torch —— api.py 模块级 `import torch`，load_audio_with_torchaudio 用 torch.Tensor
_torch_mod = types.ModuleType("torch")


class _FakeTensor:
    """极简 Tensor 替身：支持 .shape / .dim() / .mean() / .squeeze() / .to()"""

    def __init__(self, data=None, shape=None, dtype=None):
        if shape is None:
            shape = [1, 1]
        self.shape = shape
        self.dtype = dtype or "float32"
        self._data = data

    def dim(self):
        return len(self.shape)

    def mean(self, dim=None, keepdim=False):
        # 返回一个 1-D 假 tensor，足够通过后续逻辑
        return _FakeTensor(shape=[1] if not keepdim else [1, 1])

    def __getitem__(self, idx):
        return self

    def squeeze(self, dim=None):
        return self

    def to(self, *args, **kwargs):
        return self


class _FakeModule:
    """假 nn.Module 基类，不做任何事"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _FakeTensor()


_torch_mod.Tensor = _FakeTensor
_torch_mod.Module = _FakeModule
_torch_mod.nn = types.SimpleNamespace(Module=_FakeModule, functional=None)
_torch_mod.zeros = lambda *shape, **kw: _FakeTensor(shape=list(shape))
_torch_mod.tensor = lambda data, **kw: _FakeTensor(data=data)
_torch_mod.float32 = "float32"
_torch_mod.bfloat16 = "bfloat16"
_torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
_torch_mod.no_grad = lambda: __import__("contextlib").nullcontext()
# fake torch.backends（api.py 模块级设置 allow_tf32）
# 注意：torch.backends.cudnn 和 torch.backends.cuda.matmul 是独立路径
_torch_mod.backends = types.SimpleNamespace(
    cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
    cudnn=types.SimpleNamespace(allow_tf32=True),
)
# fake torch.autocast（返回 contextlib.nullcontext）
_torch_mod.autocast = lambda **kw: __import__("contextlib").nullcontext()

# fake psutil
_psutil_mod = types.ModuleType("psutil")
_psutil_mod.cpu_count = lambda logical=True: 4
_psutil_mod.virtual_memory = lambda: types.SimpleNamespace(total=8 * 1024**3, available=4 * 1024**3)

# fake prometheus_client
_prom_mod = types.ModuleType("prometheus_client")


def _noop_metric(*args, **kwargs):
    class _M:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k):
            pass

        def observe(self, *a, **k):
            pass

        def set(self, *a, **k):
            pass

    return _M()


_prom_mod.Counter = _noop_metric
_prom_mod.Histogram = _noop_metric
_prom_mod.Gauge = _noop_metric
_prom_mod.generate_latest = lambda: b""
_prom_mod.CONTENT_TYPE_LATEST = "text/plain"

# fake funasr.utils.postprocess_utils
_postprocess_mod = types.ModuleType("funasr.utils.postprocess_utils")
_postprocess_mod.rich_transcription_postprocess = lambda x: x

# fake funasr.utils
_utils_mod = types.ModuleType("funasr.utils")
_utils_mod.postprocess_utils = _postprocess_mod

# fake funasr.AutoModel —— 返回的实例需要 .generate() 方法
_funasr_mod = types.ModuleType("funasr")


class _FakeAutoModel:
    """假 AutoModel：generate 返回固定结构 [[{"text": "...", "raw_text": "..."}]]"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, input=None, **gen_kwargs):
        # SenseVoice generate 返回 list[dict]，每个 dict 含 text / raw_text
        # （见 demo1.py:41 `res[0]["text"]`）
        return [{"text": "你好世界", "raw_text": "<|zh|><|NEUTRAL|><|woitn|>你好世界"}]


_funasr_mod.AutoModel = _FakeAutoModel

# fake torchaudio —— extract_audio_metadata 用到 torchaudio.info / torchaudio.load
_torchaudio_mod = types.ModuleType("torchaudio")


class _FakeAudioInfo:
    def __init__(self, num_frames, sample_rate, num_channels=1):
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.num_channels = num_channels


class _FakeTorchaudio:
    """可控的 torchaudio 替身；测试通过设置 _next_info 控制时长"""

    def __init__(self):
        # 默认 1 秒音频 @16kHz
        self._next_info = _FakeAudioInfo(num_frames=16000, sample_rate=16000, num_channels=1)

    def info(self, file_path):
        return self._next_info

    def load(self, file_path, sample_rate=None):
        # 返回 (waveform, sr)；用 fake torch Tensor
        sr = self._next_info.sample_rate
        n = self._next_info.num_frames
        wav = _FakeTensor(shape=[1, n])
        return wav, sr


_fake_ta = _FakeTorchaudio()
_torchaudio_mod.info = _fake_ta.info
_torchaudio_mod.load = _fake_ta.load

# 注册到 sys.modules（必须在 import api 之前）
sys.modules["torch"] = _torch_mod
sys.modules["psutil"] = _psutil_mod
sys.modules["prometheus_client"] = _prom_mod
sys.modules["funasr"] = _funasr_mod
sys.modules["funasr.utils"] = _utils_mod
sys.modules["funasr.utils.postprocess_utils"] = _postprocess_mod
sys.modules["torchaudio"] = _torchaudio_mod

# 规避 _check_gpu_compatibility 在模块级抛错（api.py:279 附近）
import os
os.environ.setdefault("SENSEVOICE_DEVICE", "cpu")

# ---------------------------------------------------------------------------
# 2. import api（此时不触发真实模型加载）
# ---------------------------------------------------------------------------
import api  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# 3. 纯函数测试：取整逻辑
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "duration,expected",
    [
        (0.0, 0),       # 零
        (0.1, 1),       # 任意正小数 → 1
        (4.9, 5),       # 接近整数 → 向上
        (5.0, 5),       # 整数自身
        (5.01, 6),      # 超出一丁点 → 向上
        (6.1, 7),       # vLLM docs 示例：6.1 → 7
        (160.07, 161),  # vLLM whisper 测试用例：160.07 → 161
    ],
)
def test_seconds_uses_ceil(duration, expected):
    """usage.seconds 应为 int(math.ceil(duration))，对齐 vLLM 上游。"""
    assert int(math.ceil(duration)) == expected
    # 确保是 int 类型而非 float
    assert isinstance(int(math.ceil(duration)), int)


# ---------------------------------------------------------------------------
# 4. 端点结构测试：/api/v1/asr 与 /v1/audio/transcriptions
# ---------------------------------------------------------------------------

def _set_audio_duration(seconds: float):
    """设置 fake torchaudio.info 返回的音频时长。"""
    frames = int(seconds * 16000)
    _fake_ta._next_info = _FakeAudioInfo(
        num_frames=frames, sample_rate=16000, num_channels=1
    )


@pytest.fixture
def client():
    return TestClient(api.app)


def test_api_v1_asr_usage_seconds_ceil(client):
    """POST /api/v1/asr 的 usage.seconds 应为 ceil(总时长)。"""
    _set_audio_duration(6.1)  # 6.1 秒 → 7
    # 构造一个最小 wav 文件名上传
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    buf.seek(0)
    resp = client.post(
        "/api/v1/asr",
        files={"files": ("test.wav", buf, "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "usage" in body
    assert body["usage"]["type"] == "duration"
    assert body["usage"]["seconds"] == 7
    assert isinstance(body["usage"]["seconds"], int)


def test_v1_audio_transcriptions_usage_seconds_ceil(client):
    """POST /v1/audio/transcriptions 的 usage.seconds 应为 ceil(时长)。"""
    _set_audio_duration(5.01)  # 5.01 秒 → 6
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    buf.seek(0)
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", buf, "audio/wav")},
        data={"model": "sensevoice"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["usage"]["type"] == "duration"
    assert body["usage"]["seconds"] == 6
    assert isinstance(body["usage"]["seconds"], int)


def test_usage_seconds_zero_duration(client):
    """零时长音频 usage.seconds 应为 0，不为负、不为空。"""
    _set_audio_duration(0.0)
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"")
    buf.seek(0)
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", buf, "audio/wav")},
        data={"model": "sensevoice"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["usage"]["seconds"] == 0
    assert isinstance(body["usage"]["seconds"], int)


def test_usage_type_field_fixed(client):
    """usage.type 固定为 'duration'，与 OpenAI/vLLM 协议一致。"""
    _set_audio_duration(3.0)
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    buf.seek(0)
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", buf, "audio/wav")},
        data={"model": "sensevoice"},
    )
    assert resp.status_code == 200
    assert resp.json()["usage"]["type"] == "duration"


# ---------------------------------------------------------------------------
# 5. 新增测试：get_file_size_mb、duration 提取顺序、model 未加载 503
# ---------------------------------------------------------------------------

def test_get_file_size_mb_base64():
    """get_file_size_mb 对 base64 字符串应返回估算大小（len * 3/4）。"""
    import asyncio
    # 1MB base64 数据：解码后约 1MB，编码后 ~1.33MB
    b64_str = "A" * 1400000
    size_mb = asyncio.get_event_loop().run_until_complete(api.get_file_size_mb(b64_str))
    assert size_mb > 0
    expected = 1400000 * 3 / 4 / (1024 * 1024)
    assert abs(size_mb - expected) < 0.01


def test_get_file_size_mb_data_uri():
    """get_file_size_mb 对 data: URI base64 应返回估算大小。"""
    import asyncio
    b64_data = "A" * 1400000
    data_uri = f"data:audio/mp3;base64,{b64_data}"
    size_mb = asyncio.get_event_loop().run_until_complete(api.get_file_size_mb(data_uri))
    assert size_mb > 0
    expected = 1400000 * 3 / 4 / (1024 * 1024)
    assert abs(size_mb - expected) < 0.01


def test_get_file_size_mb_url_returns_zero():
    """get_file_size_mb 对 URL 应返回 0.0（依赖 load_audio_input_streaming 内部 HEAD 检查）。"""
    import asyncio
    size_mb = asyncio.get_event_loop().run_until_complete(
        api.get_file_size_mb("https://example.com/audio.wav")
    )
    assert size_mb == 0.0


def test_get_file_size_mb_uploadfile():
    """get_file_size_mb 对带 size 属性的 UploadFile 应返回 file.size / 1MB。"""
    import asyncio
    fake_file = types.SimpleNamespace(size=10 * 1024 * 1024)  # 10MB
    size_mb = asyncio.get_event_loop().run_until_complete(api.get_file_size_mb(fake_file))
    assert abs(size_mb - 10.0) < 0.01


def test_get_file_size_mb_bytesio():
    """get_file_size_mb 对 BytesIO 应返回 buffer 大小。"""
    import asyncio
    buf = io.BytesIO(b"x" * (5 * 1024 * 1024))  # 5MB
    size_mb = asyncio.get_event_loop().run_until_complete(api.get_file_size_mb(buf))
    assert abs(size_mb - 5.0) < 0.01


def test_duration_extracted_before_load():
    """extract_audio_metadata 在 load_audio_with_torchaudio 之前调用，duration 应正确。

    回归测试：B2/B3 修复前，audio_to_text 先 load 再 extract，指针在 EOF，
    torchaudio.info 读到空，duration=0。
    """
    _set_audio_duration(7.3)  # 7.3 秒 → ceil 8
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    buf.seek(0)
    with TestClient(api.app) as c:
        resp = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", buf, "audio/wav")},
            data={"model": "sensevoice"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 应为 ceil(7.3) = 8，不是 0
    assert body["usage"]["seconds"] == 8, f"duration 提取失败: {body}"


def test_model_not_loaded_returns_503():
    """model 未加载时端点应返回 503 而非 500。"""
    _set_audio_duration(3.0)
    original = api._model_loaded
    api._model_loaded = False
    try:
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)
        buf.seek(0)
        with TestClient(api.app) as c:
            resp = c.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", buf, "audio/wav")},
                data={"model": "sensevoice"},
            )
        assert resp.status_code == 503, f"expected 503, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "error" in body
    finally:
        api._model_loaded = original


def test_model_not_loaded_returns_503_api_v1():
    """/api/v1/asr 同样应在 model 未加载时返回 503。"""
    _set_audio_duration(3.0)
    original = api._model_loaded
    api._model_loaded = False
    try:
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)
        buf.seek(0)
        with TestClient(api.app) as c:
            resp = c.post(
                "/api/v1/asr",
                files={"files": ("test.wav", buf, "audio/wav")},
            )
        assert resp.status_code == 503, f"expected 503, got {resp.status_code}: {resp.text}"
    finally:
        api._model_loaded = original

