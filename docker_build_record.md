# SenseVoice 镜像构建记录

> 本记录跟踪所有镜像构建和推送历史

---

## 20260203-ac413e9

- **构建日期**: 2026-02-03T16:11:23+08:00
- **Git 提交**: `ac413e9732cc48900f9b531047f4eca476a35648`
- **镜像地址**: `[0;34m[INFO][0m 构建 RTX 5090 (Blackwell) 版本
[0;34m[INFO][0m 开始构建镜像: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260203-ac413e9-rtx5090
[0;34m[INFO][0m PyTorch 版本: 2.7.0
[0;34m[INFO][0m CUDA 版本: https://download.pytorch.org/whl/cu128
[0;34m[INFO][0m 构建日期: 2026-02-03T15:28:58+08:00
[0;34m[INFO][0m Git 提交: ac413e9732cc48900f9b531047f4eca476a35648
[0;31m[ERROR][0m 镜像构建失败`
- **镜像大小**: unknown
- **平台**: linux/amd64

---

## 20260203-b34ae96

- **构建日期**: 2026-02-03T10:58:14+08:00
- **Git 提交**: `b34ae969ad7a3a2af87a00b12b446b793350ef3f`
- **镜像地址**: `[0;34m[INFO][0m 构建 RTX 5090 (Blackwell) 版本
[0;34m[INFO][0m 开始构建镜像: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260203-b34ae96-rtx5090
[0;34m[INFO][0m PyTorch 版本: 2.6.0
[0;34m[INFO][0m CUDA 版本: https://download.pytorch.org/whl/cu128
[0;34m[INFO][0m 构建日期: 2026-02-03T10:57:21+08:00
[0;34m[INFO][0m Git 提交: b34ae969ad7a3a2af87a00b12b446b793350ef3f
[0;31m[ERROR][0m 镜像构建失败`
- **镜像大小**: unknown
- **平台**: linux/amd64

---

## 20260129-4e73b87

- **构建日期**: 2026-01-29T18:22:52+08:00
- **Git 提交**: `4e73b87fcbd6a6acbb811fc100695a15da8ea18e`
- **镜像地址**: `[0;34m[INFO][0m 开始构建镜像: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260129-4e73b87
[0;34m[INFO][0m 构建日期: 2026-01-29T18:22:47+08:00
[0;34m[INFO][0m Git 提交: 4e73b87fcbd6a6acbb811fc100695a15da8ea18e
[0;32m[SUCCESS][0m 镜像构建并推送成功: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260129-4e73b87
hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260129-4e73b87`
- **镜像大小**: unknown
- **平台**: linux/amd64

---

## 20260129-df45952

- **构建日期**: 2026-01-29T18:05:07+08:00
- **Git 提交**: `df45952952a0a89cb1b0d967ddc5893b497b77e5`
- **镜像地址**: `[0;34m[INFO][0m 开始构建镜像: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260129-df45952
[0;34m[INFO][0m 构建日期: 2026-01-29T18:05:01+08:00
[0;34m[INFO][0m Git 提交: df45952952a0a89cb1b0d967ddc5893b497b77e5
[0;32m[SUCCESS][0m 镜像构建并推送成功: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260129-df45952
hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260129-df45952`
- **镜像大小**: unknown
- **平台**: linux/amd64

---

## 20260128-16cc717

- **构建日期**: 2026-01-28T15:06:39+08:00
- **Git 提交**: `16cc71791c3e4d3b809490fe0fddf370855b6c7e`
- **镜像地址**: `[0;34m[INFO][0m 开始构建镜像: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260128-16cc717
[0;34m[INFO][0m 构建日期: 2026-01-28T14:07:41+08:00
[0;34m[INFO][0m Git 提交: 16cc71791c3e4d3b809490fe0fddf370855b6c7e
[0;32m[SUCCESS][0m 镜像构建并推送成功: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260128-16cc717
hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260128-16cc717`
- **镜像大小**: unknown
- **平台**: linux/amd64

---

## 20260128-8cc68fb

- **构建日期**: 2026-01-28T11:52:08+08:00
- **Git 提交**: `8cc68fb04a2062187b947baec10d3dd5baf3d57f`
- **镜像地址**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260128-8cc68fb`
- **镜像大小**: 6.08GB
- **平台**: linux/amd64
- **Bug 修复**:
  - 修复 `siliconflow_transcribe` BytesIO 指针在 `extract_audio_metadata` 后未重置问题 (P0 Bug)

---

## 20260127-1f26442

- **构建日期**: 2026-01-27T21:03:26+08:00
- **Git 提交**: `1f26442ea1fa3565714084e53081f3bbc5ae14e6`
- **镜像地址**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260127-1f26442`
- **镜像大小**: 6.08GB
- **平台**: linux/amd64
- **Bug 修复**:
  - 修复 `siliconflow_transcribe` 重复读取文件导致 UploadFile 指针失效 (P0 Bug)
  - 修复 `/ready` 端点 GPU 检查可能崩溃问题
  - 修复 `audio_to_text` 缺少模型加载保护问题
  - 修复 `webui.py` 变量 `model` 被覆盖问题
  - 修复 `webui.py` `event_set` 末尾多余逗号

---

## 20260128-cf2f9bf

- **构建日期**: 2026-01-28
- **Git 提交**: `cf2f9bfad1f68a1c389d33a0501b0a4e53431941`
- **镜像地址**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260128-cf2f9bf`
- **镜像标签**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:latest`
- **镜像大小**: 6.08GB
- **平台**: linux/amd64
- **Bug 修复**:
  - 添加缺失的 `psutil` 和 `httpx` 依赖到 requirements.txt

---

## 使用说明

### 快速构建推送

```bash
# 构建并推送新镜像 (自动生成版本标签)
./build_and_push.sh

# 使用指定版本标签
./build_and_push.sh v1.0.0

# 跳过登录步骤
./build_and_push.sh --no-login
```

### 版本标签规则

- **自动生成**: `YYYYMMDD-gittag` (如 `20250123-v1.0.0`)
- **不使用 latest 标签**，确保每次构建有唯一标识

### SiliconFlow 镜像仓库

- **仓库地址**: `hub.6scloud.com`
- **命名空间**: `clxuaivn500083i7ncuxpw8cf`
- **镜像名称**: `sensevoice`
- **完整地址**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:{tag}`

---

> 历史版本记录将保留在此处