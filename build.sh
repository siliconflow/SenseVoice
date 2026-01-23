#!/bin/bash
#
# SenseVoice 镜像构建与推送脚本
# 遵守项目规范，使用语义化版本标签
#
# 使用方法:
#   ./build.sh                    # 构建最新版本
#   ./build.sh --push             # 构建并推送
#   ./build.sh --push --latest    # 构建并推送 latest 标签
#   ./build.sh v1.0.0             # 构建指定版本
#

set -e

# 配置
IMAGE_NAME="sensevoice-api"
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
ORG="${DOCKER_ORG:-sensevoice}"

# 解析参数
VERSION=""
PUSH=false
LATEST=false
BUILD_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --latest)
            LATEST=true
            shift
            ;;
        --no-cache)
            BUILD_ARGS="--no-cache $BUILD_ARGS"
            shift
            ;;
        --platform)
            BUILD_ARGS="$BUILD_ARGS --platform $2"
            shift 2
            ;;
        v*)
            VERSION="$1"
            shift
            ;;
        -*)
            echo "未知参数: $1"
            exit 1
            ;;
        *)
            shift
            ;;
    esac
done

# 获取版本号
if [[ -z "$VERSION" ]]; then
    # 从 git 获取版本（遵循语义化版本）
    # 尝试从 git tag 获取，否则使用 commit hash
    VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    if [[ -z "$VERSION" ]]; then
        # 使用 commit hash 短版本
        VERSION="dev-$(git rev-parse --short HEAD)"
    fi
fi

# 标准化版本号（去掉 v 前缀用于标签）
VERSION_TAG="${VERSION#v}"
FULL_IMAGE="${REGISTRY}/${ORG}/${IMAGE_NAME}"

echo "============================================"
echo "SenseVoice 镜像构建脚本"
echo "============================================"
echo "镜像名称: ${FULL_IMAGE}"
echo "版本标签: ${VERSION_TAG}"
echo "推送镜像: ${PUSH}"
echo "打 latest: ${LATEST}"
echo "构建参数: ${BUILD_ARGS:-无}"
echo "============================================"

# 检查是否有未提交的更改
if [[ -n "$(git status --porcelain)" ]]; then
    echo "⚠️  警告: 有未提交的代码更改"
    git status --short
    echo ""
fi

# 检查 Docker 是否可用
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker 未安装或不可用"
    exit 1
fi

# 构建镜像
echo ""
echo "🚀 开始构建镜像..."
echo ""

docker build \
    -t "${FULL_IMAGE}:${VERSION_TAG}" \
    $BUILD_ARGS \
    .

if [[ $? -ne 0 ]]; then
    echo "❌ 构建失败"
    exit 1
fi

echo ""
echo "✅ 镜像构建成功: ${FULL_IMAGE}:${VERSION_TAG}"

# 打 latest 标签
if [[ "$LATEST" == true ]]; then
    docker tag "${FULL_IMAGE}:${VERSION_TAG}" "${FULL_IMAGE}:latest"
    echo "✅ 已打 latest 标签: ${FULL_IMAGE}:latest"
fi

# 推送镜像
if [[ "$PUSH" == true ]]; then
    echo ""
    echo "📤 推送镜像..."

    # 推送版本标签
    docker push "${FULL_IMAGE}:${VERSION_TAG}"
    echo "✅ 已推送: ${FULL_IMAGE}:${VERSION_TAG}"

    # 推送 latest 标签
    if [[ "$LATEST" == true ]]; then
        docker push "${FULL_IMAGE}:latest"
        echo "✅ 已推送: ${FULL_IMAGE}:latest"
    fi

    echo ""
    echo "🎉 镜像推送完成!"
    echo ""
    echo "使用方式:"
    echo "  docker pull ${FULL_IMAGE}:${VERSION_TAG}"
    echo "  docker run -d --gpus all -p 8000:8000 ${FULL_IMAGE}:${VERSION_TAG}"
else
    echo ""
    echo "📋 下一步操作:"
    echo "  # 推送镜像"
    echo "  docker push ${FULL_IMAGE}:${VERSION_TAG}"
    echo ""
    echo "  # 或打 latest 并推送"
    echo "  docker tag ${FULL_IMAGE}:${VERSION_TAG} ${FULL_IMAGE}:latest"
    echo "  docker push ${FULL_IMAGE}:${VERSION_TAG}"
fi

echo ""
echo "============================================"