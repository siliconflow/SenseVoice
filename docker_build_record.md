# SenseVoice 镜像构建记录

> 本记录跟踪所有镜像构建和推送历史

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

## 20260127-cf2f9bf

- **构建日期**: 2026-01-27
- **Git 提交**: `cf2f9bfad1f68a1c389d33a0501b0a4e53431941`
- **镜像地址**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260127-cf2f9bf`
- **镜像标签**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:latest`
- **镜像大小**: 6.08GB
- **平台**: linux/amd64
- **变更内容**:
  - 缓存文件清理：延迟清理临时音频文件，支持环境变量配置
  - 请求信息记录：记录接口、trace id、音频元信息
  - 健康检查：增加 `/health` 和 `/ready` 接口
  - API 文档更新
  - 标点控制：支持环境变量 `SENSEVOICE_USE_PUNC` 和 `SENSEVOICE_PUNC_MODEL`

---

## 20260126-d2d4a64

- **构建日期**: 2026-01-26T20:39:26+08:00
- **Git 提交**: `d2d4a64d5da3f8f296d82e554a0d496ed24a7385`
- **镜像地址**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260126-d2d4a64`
- **镜像大小**: unknown
- **平台**: linux/amd64

---

## 20260123-8eb7f1a

- **构建日期**: 2026-01-23T19:10:32+08:00
- **Git 提交**: `8eb7f1a358370a94836a5d4f6b7a34dc5df428de`
- **镜像地址**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260123-8eb7f1a`
- **镜像大小**: unknown
- **平台**: linux/amd64

---

## 20260123-011181d

- **构建日期**: 2026-01-23T20:00:00+08:00
- **Git 提交**: `011181de0632efac34e95707e069e8eb2797b8ab`
- **镜像地址**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260123-011181d`
- **镜像大小**: 6.08GB
- **平台**: linux/amd64
- **修复**: 修复 api.py/demo2.py/export.py 的 `from model import SenseVoiceSmall` 导入错误

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

## 构建版本记录

### v0.0.1

- **构建日期**: 2026-01-23
- **Git 提交**: 0c0e606 (已完成镜像构建、推送 Skill 和记录系统的建立)
- **镜像地址**: `hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260123-v0.0.1`
- **平台**: linux/amd64
- **优化**: 多阶段构建，Python 虚拟环境，非 root 用户

---

> 历史版本记录将保留在此处