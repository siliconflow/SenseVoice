#!/bin/bash
#
# 模型预下载脚本
# 用于在构建镜像前或启动容器前下载模型权重
# 利用 SiliconFlow 缓存加速（如果可用）
#
# 使用方法:
#   chmod +x download_model.sh
#   ./download_model.sh                    # 下载默认模型到 ./models 目录
#   ./download_model.sh /path/to/cache     # 下载到指定目录
#
# 环境变量:
#   MODEL_NAME     - 模型名称 (默认: iic/SenseVoiceSmall)
#   MODELSCOPE_CACHE - ModelScope 缓存目录

set -e

# 配置
DEFAULT_MODEL_NAME="iic/SenseVoiceSmall"
CACHE_DIR="${1:-./models}"

echo "========================================"
echo "SenseVoice 模型预下载脚本"
echo "========================================"
echo "模型名称: ${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
echo "缓存目录: $CACHE_DIR"
echo "========================================"

# 创建缓存目录
mkdir -p "$CACHE_DIR"

# 使用 Python 下载模型
python3 -c "
import os
import sys

# 设置环境变量
os.environ['MODELSCOPE_CACHE'] = '$CACHE_DIR'
os.environ['HF_HOME'] = '$CACHE_DIR/.cache'

print('正在从 ModelScope 下载模型...')

from modelscope import snapshot_download

model_name = os.environ.get('MODEL_NAME', '$DEFAULT_MODEL_NAME')
print(f'开始下载模型: {model_name}')

# 下载模型文件
model_path = snapshot_download(
    model_name,
    cache_dir='$CACHE_DIR',
    allow_file_pattern=['*.bin', '*.safetensors', '*.pt', '*.py', '*.json'],
)

print(f'模型下载完成: {model_path}')

# 显示下载的文件
print()
print('已下载的文件:')
for root, dirs, files in os.walk('$CACHE_DIR'):
    for f in files:
        fpath = os.path.join(root, f)
        size = os.path.getsize(fpath)
        if size > 1024*1024:
            print(f'  {os.path.relpath(fpath, \"$CACHE_DIR\")}: {size/1024/1024:.1f} MB')
        elif size > 1024:
            print(f'  {os.path.relpath(fpath, \"$CACHE_DIR\")}: {size/1024:.1f} KB')
"