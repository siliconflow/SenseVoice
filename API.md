# SenseVoice API 文档

SenseVoice 提供 REST API 服务，支持语音识别（ASR）、语音情感识别（SER）和音频事件检测（AED）。

## 启动服务

```bash
python api.py
```

服务默认运行在 `http://0.0.0.0:50000`

## 端点列表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | API 主页 |
| POST | `/api/v1/asr` | 语音识别主接口 |

## API 详解

### POST /api/v1/asr

音频转文字接口。

**请求格式:** `multipart/form-data`

**请求参数:**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `files` | List[File] | 是 | 音频文件列表（支持 wav、mp3，采样率 16kHz） |
| `keys` | str | 否 | 音频名称列表，用逗号分隔，用于标识返回结果 |
| `lang` | str | 否 | 音频语言，支持: `auto`(自动检测), `zh`(中文), `en`(英文), `yue`(粤语), `ja`(日语), `ko`(韩语), `nospeech`(无语音) |

**响应格式:**

```json
{
  "result": [
    {
      "raw_text": "<|startoftranscript|><|zh|><|neutral|><|woitn|><|Text|>",
      "clean_text": "这是一段测试文本",
      "text": "这是一段测试文本。"
    }
  ]
}
```

**响应字段说明:**

| 字段 | 类型 | 描述 |
|------|------|------|
| `raw_text` | str | 包含特殊标记的原始输出 |
| `clean_text` | str | 去除特殊标记后的纯文本 |
| `text` | str | 经过后处理的文本，包含标点符号 |

## 使用示例

### cURL

```bash
# 单个音频文件
curl -X POST "http://localhost:50000/api/v1/asr" \
  -F "files=@audio.wav" \
  -F "lang=zh"

# 多个音频文件
curl -X POST "http://localhost:50000/api/v1/asr" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "keys=audio1,audio2" \
  -F "lang=auto"
```

### Python (requests)

```python
import requests

url = "http://localhost:50000/api/v1/asr"

# 单个文件
with open("audio.wav", "rb") as f:
    files = {"files": ("audio.wav", f, "audio/wav")}
    data = {"lang": "zh"}
    response = requests.post(url, files=files, data=data)
    print(response.json())

# 多个文件
with open("audio1.wav", "rb") as f1, open("audio2.wav", "rb") as f2:
    files = [
        ("files", ("audio1.wav", f1, "audio/wav")),
        ("files", ("audio2.wav", f2, "audio/wav")),
    ]
    data = {"keys": "audio1,audio2", "lang": "auto"}
    response = requests.post(url, files=files, data=data)
    print(response.json())
```

### JavaScript (fetch)

```javascript
async function transcribeAudio(filePath, lang = 'auto') {
  const formData = new FormData();
  const file = new File([await fetch(filePath).then(r => r.blob())], 'audio.wav');
  formData.append('files', file);
  formData.append('lang', lang);

  const response = await fetch('http://localhost:50000/api/v1/asr', {
    method: 'POST',
    body: formData
  });
  return response.json();
}

// 使用
transcribeAudio('audio.wav', 'zh').then(console.log);
```

## 特殊标记说明

模型输出包含以下特殊标记:

```
<|startoftranscript|>  - 起始标记
<|zh|><|en|><|yue|><|ja|><|ko|><|nospeech|> - 语言标识
<|happy|><|sad|><|angry|><|neutral|><|unk|> - 情感标识
<|withitn|><|woitn|> - 文本norm标识
<|Text|> - 识别文本
```

## 环境配置

| 环境变量 | 默认值 | 描述 |
|----------|--------|------|
| `SENSEVOICE_DEVICE` | `cuda:0` | 运行设备，可设为 `cpu`、`cuda:0` 等 |

## 错误处理

```json
{
  "detail": "错误描述"
}
```

## 交互式文档

启动服务后，访问 `http://localhost:50000/docs` 可查看 Swagger 交互式文档。

## 注意事项

1. 音频建议采样率为 16kHz，服务会自动重采样
2. 每次请求最多支持多个音频文件
3. `lang` 参数设为 `auto` 时自动检测语言
4. 实际使用时建议将 `use_itn` 设为 `true` 以获得更好的文本效果（如需要可在 `api.py` 中修改）