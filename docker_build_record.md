# SenseVoice 镜像构建记录

> 本记录跟踪所有镜像构建和推送历史

---

## 20260123-8eb7f1a

- **构建日期**: 2026-01-23T19:10:32+08:00
- **Git 提交**: `8eb7f1a358370a94836a5d4f6b7a34dc5df428de`
- **镜像地址**: `[0;34m[INFO][0m 开始构建镜像: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260123-8eb7f1a
[0;34m[INFO][0m 构建日期: 2026-01-23T18:45:37+08:00
[0;34m[INFO][0m Git 提交: 8eb7f1a358370a94836a5d4f6b7a34dc5df428de
[0;32m[SUCCESS][0m 镜像构建并推送成功: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260123-8eb7f1a
hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260123-8eb7f1a`
- **镜像大小**: unknown
- **平台**: linux/amd64

---

## 20260123-8eb7f1a

- **构建日期**: 2026-01-23T18:44:02+08:00
- **Git 提交**: `8eb7f1a358370a94836a5d4f6b7a34dc5df428de`
- **镜像地址**: `[0;34m[INFO][0m 开始构建镜像: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260123-8eb7f1a
[0;34m[INFO][0m 构建日期: 2026-01-23T18:44:00+08:00
[0;34m[INFO][0m Git 提交: 8eb7f1a358370a94836a5d4f6b7a34dc5df428de
[0;31m[ERROR][0m 镜像构建失败`
- **镜像大小**: unknown
- **平台**: linux/amd64

---

## 20260123-8eb7f1a

- **构建日期**: 2026-01-23T18:43:44+08:00
- **Git 提交**: `8eb7f1a358370a94836a5d4f6b7a34dc5df428de`
- **镜像地址**: `[0;34m[INFO][0m 开始构建镜像: hub.6scloud.com/clxuaivn500083i7ncuxpw8cf/sensevoice:20260123-8eb7f1a
[0;34m[INFO][0m 构建日期: 2026-01-23T18:43:34+08:00
[0;34m[INFO][0m Git 提交: 8eb7f1a358370a94836a5d4f6b7a34dc5df428de
[0;31m[ERROR][0m 镜像构建失败`
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