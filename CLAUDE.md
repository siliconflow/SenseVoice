# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SenseVoice** - Alibaba's multimodal speech recognition and understanding model. Provides ASR (automatic speech recognition), SER (speech emotion recognition), and AED (audio event detection) in a single model.

**Key Capabilities:**
- Languages: Chinese (zh), English (en), Japanese (ja), Korean (ko), Cantonese (yue)
- Emotion detection: happy, sad, angry, neutral, unknown
- Audio event detection
- Inverse text normalization (ITN)
- Timestamp output support

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Web interface
python webui.py

# REST API server
python api.py

# Basic inference
python demo1.py --file <audio_file>

# With language specification
python demo1.py --file audio.wav --language zh
```

## Architecture

### Core Model (`model.py`)

**SenseVoiceEncoderSmall** (lines 437-577):
- Streaming chunk-aware multihead attention encoder (SCAMA)
- Uses `MultiHeadedAttentionSANM` with FSMN memory block
- Configurable: `num_blocks`, `attention_heads`, `linear_units`, `kernel_size`

**SenseVoiceSmall** (lines 580-930):
- CTC-attention hybrid Encoder-Decoder
- `forward()` - Training forward pass with CTC + CE loss
- `inference()` - Inference with CTC decoding
- `encode()` - Encodes speech with language/style query injection

### Key Components

- **MultiHeadedAttentionSANM** (lines 74-265): Attention with FSMN memory, supports chunk-based streaming inference
- **EncoderLayerSANM** (lines 294-434): Encoder layer with stochastic depth support
- **LayerNorm** (lines 268-280): FP32-safe layer normalization

### Special Tokens

The model uses fixed special tokens in the output:
- Position 0: `<|startoftranscript|>`
- Position 1: Language ID (e.g., `<|zh|>`, `<|en|>`, `<|ja|>`, `<|yue|>`)
- Position 2: Emotion ID (e.g., `<|neutral|>`)
- Position 3: Text norm flag (`<|withitn|>` or `<|woitn|>`)
- Position 4+: Decoded text

### Language/Emotion Dictionaries

```python
lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
textnorm_dict = {"withitn": 14, "woitn": 15}
emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}
```

## Inference Entry Points

| File | Purpose |
|------|---------|
| `demo1.py` | Basic inference with FunASR AutoModel wrapper |
| `demo2.py` | Direct model instantiation |
| `api.py` | FastAPI server (`POST /recognition`) |
| `webui.py` | Gradio web interface |

### APIs

**REST API (`api.py`):**
- `POST /recognition` - Audio transcription
- Supports: file upload, language selection, ITN toggle, timestamp output

**Frontend Utilities (`utils/frontend.py`):**
- Handles audio loading, feature extraction (FBank), data augmentation

## Model Export

```bash
# PyTorch exported model
python export.py --model iic/SenseVoiceSmall --export_dir ./export

# ONNX
python demo_onnx.py

# LibTorch
python demo_libtorch.py
```

## Fine-tuning

```bash
# Using DeepSpeed (see finetune.sh)
./finetune.sh

# Key configuration (from finetune.sh):
# - torchrun with DISTRIBUTED_ARGS
# - train_ds.py from funasr package
# - Token-based batch sampling (batch_size=6000)
# - LR: 0.0002
# - Save every 2000 steps
```

## Important Implementation Details

1. **Input preprocessing**: Speech is prepended with language query, emotion query, and style query embeddings (3 tokens added to sequence)

2. **CTC decoding**: Greedy decoding with duplicate token removal (`torch.unique_consecutive`)

3. **Chunk-based streaming**: `forward_chunk()` methods support cache-based inference for streaming audio

4. **Loss computation**: Two parallel losses - CTC loss on main text, CE loss on first 4 tokens (language/style/emo)

5. **FP16 safety**: `LayerNorm.forward()` casts to float32 before normalization to prevent FP16 instability

## Dependencies

Key packages from `requirements.txt`:
- `funasr` - Core speech recognition framework
- `torch` >= 2.0.0
- `gradio` >= 4.0.0 - Web UI
- `fastapi` + `uvicorn` - REST API
- `modelscope` - Model loading

## Debugging Tips

- Check `model.inference()` meta_data for timing breakdown (load_data, extract_feat)
- Use `output_timestamp=True` in inference for word-level timestamps
- Set `SENSEVOICE_DEVICE=cuda` or `cpu` to override device selection