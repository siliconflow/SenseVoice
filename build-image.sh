#!/bin/bash
# SenseVoice 统一构建脚本
# 强制规则：每次构建前必须递增 VERSION 文件，否则构建失败
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VERSION_FILE="$PROJECT_DIR/VERSION"

REGISTRY="${REGISTRY:-hub.6scloud.com}"
NAMESPACE="${NAMESPACE:-clxuaivn500083i7ncuxpw8cf}"
PROJECT_NAME="sensevoice"
PLATFORM="${PLATFORM:-linux/amd64}"

# 读取版本号
if [[ ! -f "$VERSION_FILE" ]]; then
    echo "ERROR: VERSION file not found at $VERSION_FILE"
    echo "Please create it with the next version number (e.g., 1.0.1)"
    exit 1
fi
VERSION=$(tr -d '[:space:]' < "$VERSION_FILE")
DATE_TAG=$(date +%Y%m%d)
BASE_TAG="${DATE_TAG}-v${VERSION}"

echo "========================================"
echo "SenseVoice Build"
echo "Version : $VERSION"
echo "Base tag: $BASE_TAG"
echo "Registry: $REGISTRY/$NAMESPACE/$PROJECT_NAME"
echo "========================================"
echo ""

# 检查本地/远程是否已有相同 tag 的镜像
check_existing() {
    local gpu=$1
    local full_tag="${BASE_TAG}-${gpu}"
    local full_image="${REGISTRY}/${NAMESPACE}/${PROJECT_NAME}:${full_tag}"
    if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${full_image}$"; then
        echo "ERROR: Image already exists locally: $full_image"
        echo "Rule: Every build MUST have a unique tag."
        echo "Fix: Edit $VERSION_FILE and bump the version number before rebuilding."
        exit 1
    fi
    if command -v skopeo &>/dev/null; then
        if skopeo inspect "docker://${full_image}" &>/dev/null; then
            echo "ERROR: Image already exists in registry: $full_image"
            echo "Fix: Edit $VERSION_FILE and bump the version number."
            exit 1
        fi
    fi
}

# 构建并推送指定 GPU 的镜像
build_and_push() {
    local gpu=$1
    local torch_version=$2
    local torch_index=$3
    local full_tag="${BASE_TAG}-${gpu}"
    local full_image="${REGISTRY}/${NAMESPACE}/${PROJECT_NAME}:${full_tag}"

    echo ">>> Building $gpu image"
    echo "    PyTorch : $torch_version"
    echo "    CUDA    : $torch_index"
    echo "    Tag     : $full_tag"

    docker buildx build \
        --platform "$PLATFORM" \
        --build-arg TORCH_VERSION="$torch_version" \
        --build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/${torch_index}" \
        -t "$full_image" \
        -f "$PROJECT_DIR/Dockerfile" \
        "$PROJECT_DIR"

    echo ">>> Pushing $full_image"
    docker push "$full_image"
    echo "    Done: $full_image"
    echo ""
}

# 更新 deployment yaml 中的 image 字段
update_deployment() {
    local yaml_file=$1
    local gpu=$2
    local full_tag="${BASE_TAG}-${gpu}"
    local full_image="${REGISTRY}/${NAMESPACE}/${PROJECT_NAME}:${full_tag}"

    if [[ -f "$yaml_file" ]]; then
        echo ">>> Updating $(basename "$yaml_file")"
        sed -i "/image: .*\/${PROJECT_NAME}:/c\\        image: ${full_image}" "$yaml_file"
        echo "    -> $full_image"
    else
        echo "WARNING: $yaml_file not found, skipping"
    fi
}

# 主流程
echo "Checking for existing tags..."
check_existing "rtx4090"
check_existing "rtx5090"
echo "OK — tags are unique."
echo ""

build_and_push "rtx4090" "2.5.1" "cu124"
build_and_push "rtx5090" "2.7.0" "cu128"

echo ">>> Updating deployment configs..."
update_deployment "$PROJECT_DIR/k8s-deployment.yaml" "rtx4090"
echo ""

echo "========================================"
echo "Build completed successfully!"
echo ""
echo "Images:"
echo "  ${REGISTRY}/${NAMESPACE}/${PROJECT_NAME}:${BASE_TAG}-rtx4090"
echo "  ${REGISTRY}/${NAMESPACE}/${PROJECT_NAME}:${BASE_TAG}-rtx5090"
echo ""
echo "Next steps:"
echo "  1. git diff 检查 k8s-deployment.yaml 变更"
echo "  2. git commit -am \"Bump VERSION to $VERSION, rebuild images\""
echo "  3. 部署更新后的 k8s-deployment.yaml"
echo "  4. 编辑 VERSION 文件，为下次构建做准备"
echo "========================================"
