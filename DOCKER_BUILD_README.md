# SenseVoice Docker 构建指南

## 概述

本项目包含完整的 Docker 镜像构建、推送和记录系统，支持在 Mac Apple Silicon 上构建 linux/amd64 镜像并推送到 SiliconFlow 镜像仓库。

## 目录结构

```
SenseVoice/
├── Dockerfile              # Docker 镜像定义 (多阶段构建优化版)
├── build_and_push.sh       # 镜像构建推送脚本
├── docker_build_record.md  # 构建记录文件
├── docker-compose.yaml     # Docker Compose 配置
└── DOCKER_BUILD_README.md  # 本文档
```

## 快速开始

### 1. 构建并推送镜像

```bash
# 赋予执行权限 (首次使用)
chmod +x build_and_push.sh

# 自动生成版本标签并推送
./build_and_push.sh

# 使用指定版本标签
./build_and_push.sh v1.0.0
```

### 2. 从 SiliconFlow 拉取使用

```bash
# 拉取镜像
docker pull hub.6scloud.com/d1r7umcsfi9c73b4drdg/sensevoice:{version_tag}

# 运行容器 (支持 GPU)
docker run -d \
    --name sensevoice \
    -p 8000:8000 \
    -e SENSEVOICE_DEVICE=cuda \
    hub.6scloud.com/d1r7umcsfi9c73b4drdg/sensevoice:{version_tag}
```

### 3. 使用 Docker Compose

```bash
# 启动服务
SENSEVOICE_DEVICE=cuda API_PORT=8000 docker-compose up -d

# 查看日志
docker-compose logs -f sensevoice-api
```

## 镜像特点

### 优化策略

1. **多阶段构建**: 分离编译环境和运行环境
2. **Python 虚拟环境**: 隔离依赖，不污染系统
3. **最小化基础镜像**: 使用 `python:3.10-slim-bookworm`
4. **非 root 用户**: 增强安全性
5. **健康检查**: 内置容器健康检测

### 镜像大小

预期镜像大小: **~3-4 GB** (含 PyTorch/FunASR)

## 版本标签规则

- **自动生成**: `YYYYMMDD-{git_tag}` (如 `20250123-v1.0.0`)
- **禁止使用 latest 标签**，确保版本可追溯

## 构建记录

每次构建推送后，构建记录会自动更新到 `docker_build_record.md`，包含:
- 构建日期
- Git 提交 ID
- 镜像完整地址
- 镜像大小

## 故障排除

### Docker Buildx 未安装

```bash
# 安装 Docker Buildx
docker buildx install
```

### 构建平台错误

确保使用 `--platform linux/amd64` 参数，脚本已自动处理。

### 推送失败

检查登录状态:
```bash
docker login hub.6scloud.com
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SENSEVOICE_DEVICE` | 设备选择 | `cuda` |
| `PYTHONUNBUFFERED` | Python 输出不缓冲 | `1` |
| `OMP_NUM_THREADS` | OpenMP 线程数 | `4` |
| `MODELSCOPE_CACHE` | 模型缓存目录 | `/models` |
| `TEMP_DIR` | 临时文件目录 | `/tmp/sensevoice` |

## 端口

| 端口 | 说明 |
|------|------|
| 8000 | API 服务端口 |

## 安全说明

- 容器内使用非 root 用户 `appuser`
- 临时文件和日志目录有独立权限
- 健康检查使用 `curl` 而非 root 权限命令