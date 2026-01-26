# SiliconFlow API 兼容性评估报告

## 概述

当前 SenseVoice 实现的 API 与 SiliconFlow 标准 API 存在多处不兼容，需调整方可对接。

## SiliconFlow 标准接口

```
POST https://api.siliconflow.cn/v1/audio/transcriptions
Authorization: Bearer <API_KEY>
```

**请求参数:**
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| file | binary | 是 | 音频文件（≤50MB） |
| model | string | 是 | 模型名称，如 `FunAudioLLM/SenseVoiceSmall` |

**响应格式:**
```json
{ "text": "转录文本内容" }
```

## 当前实现

```
POST http://localhost:50000/api/v1/asr
认证: 无
```

**请求参数:**
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| files | List[File] | 是 | 音频文件列表 |
| keys | string | 否 | 文件标识列表 |
| lang | string | 否 | 语言类型 |

**响应格式:**
```json
{
  "result": [
    { "raw_text": "...", "clean_text": "...", "text": "..." }
  ]
}
```

## 差异对比

| 对比项 | SiliconFlow | 当前实现 | 兼容性 |
|--------|-------------|----------|--------|
| 端点路径 | `/v1/audio/transcriptions` | `/api/v1/asr` | ❌ |
| 认证方式 | Bearer Token | 无 | ❌ |
| 模型参数 | `model` 必需 | 固定 `iic/SenseVoiceSmall` | ❌ |
| 文件参数 | 单文件 `file` | 多文件 `files` | ❌ |
| 响应结构 | `{"text": "..."}` | `{"result": [...]}` | ❌ |

## 兼容性评分: 1/10

## 适配建议

如需兼容 SiliconFlow API，需进行以下改造：

### 1. 添加认证中间件
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    # 验证 token 逻辑
```

### 2. 添加模型参数
```python
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("FunAudioLLM/SenseVoiceSmall"),
    # ...
):
    pass
```

### 3. 调整响应格式
```python
return {"text": result[0]["text"]}
```

### 4. 兼容模式示例

```python
@app.post("/v1/audio/transcriptions")
async def siliconflow_compatible(
    file: UploadFile = File(...),
    model: str = Form("FunAudioLLM/SenseVoiceSmall"),
    language: str = Form(None),
):
    # 兼容原接口
    files = [file]
    lang = language or "auto"
    # ... 处理逻辑
    return {"text": clean_text}
```

## 最小改动方案

保持现有 `/api/v1/asr` 接口不变，额外暴露 SiliconFlow 兼容接口：

```python
@app.post("/v1/audio/transcriptions", response_model=SiliconFlowResponse)
async def siliconflow_transcribe(
    file: UploadFile = File(...),
    model: str = Form("FunAudioLLM/SenseVoiceSmall"),
):
    """SiliconFlow 兼容接口"""
    # 读取文件 -> 识别 -> 返回
    return SiliconFlowText(text=clean_text)
```

## 结论

当前 API 设计理念与 SiliconFlow 不同：
- SiliconFlow 走标准 OpenAI 兼容风格
- 当前实现更接近 FunASR 原始风格

建议：新增兼容端点而非改造现有接口，保持向后兼容。