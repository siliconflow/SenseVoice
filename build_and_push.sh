#!/bin/bash
# ==================== SenseVoice 镜像构建推送脚本 ====================
# 用于构建 linux/amd64 镜像并推送到 SiliconFlow 镜像仓库

set -e

# 配置变量 (override via environment variables)
REGISTRY="${REGISTRY:-hub.6scloud.com}"
NAMESPACE="${NAMESPACE:-your-namespace}"
IMAGE_NAME="${IMAGE_NAME:-sensevoice}"
PLATFORM="${PLATFORM:-linux/amd64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_RECORD="${SCRIPT_DIR}/docker_build_record.md"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 获取版本标签
get_version_tag() {
    local date_tag=$(date +%Y%m%d)
    local git_tag=$(git describe --tags --always 2>/dev/null || echo "unknown")
    echo "${date_tag}-${git_tag}"
}

# 检查 Docker Buildx
check_buildx() {
    if ! docker buildx version &>/dev/null; then
        log_error "Docker Buildx 未安装，请先安装 Docker Buildx"
        exit 1
    fi
    log_success "Docker Buildx 已就绪"
}

# 创建 builder 实例 (如果不存在)
setup_builder() {
    if ! docker buildx inspect sensevoice-builder &>/dev/null; then
        log_info "创建新的 builder 实例..."
        docker buildx create \
            --name sensevoice-builder \
            --driver docker-container \
            --use
    else
        log_info "使用现有的 builder 实例..."
        docker buildx use sensevoice-builder
    fi
    docker buildx inspect --bootstrap
}

# 构建镜像
build_image() {
    local version_tag=$1
    local target_arch=${2:-"rtx4090"}  # 默认构建 RTX 4090 版本
    local full_image="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${version_tag}"
    local build_date=$(date -Iseconds)
    local git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    # 根据目标架构选择 PyTorch 版本
    # PyTorch 版本说明:
    # - CUDA 12.4 (RTX 4090): PyTorch 2.5.1+cu124
    # - CUDA 12.8 (RTX 5090): PyTorch 2.7.0+cu128 (2.6.0 在 cu128 中不可用，使用 2.7.0)
    local torch_version="2.5.1"
    local torch_index_url="https://download.pytorch.org/whl/cu124"

    if [ "${target_arch}" = "rtx5090" ]; then
        torch_version="2.7.0"
        torch_index_url="https://download.pytorch.org/whl/cu128"
        full_image="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${version_tag}-rtx5090"
        log_info "构建 RTX 5090 (Blackwell) 版本"
    else
        log_info "构建 RTX 4090 (Ada) 版本"
    fi

    log_info "开始构建镜像: ${full_image}"
    log_info "PyTorch 版本: ${torch_version}"
    log_info "CUDA 版本: ${torch_index_url}"
    log_info "构建日期: ${build_date}"
    log_info "Git 提交: ${git_commit}"

    # 构建 linux/amd64 镜像
    docker buildx build \
        --platform ${PLATFORM} \
        --tag "${full_image}" \
        --push \
        --build-arg BUILD_DATE="${build_date}" \
        --build-arg VCS_REF="${git_commit}" \
        --build-arg VERSION="${version_tag}" \
        --build-arg TORCH_VERSION="${torch_version}" \
        --build-arg TORCH_INDEX_URL="${torch_index_url}" \
        . || {
            log_error "镜像构建失败"
            return 1
        }

    log_success "镜像构建并推送成功: ${full_image}"
    echo "${full_image}"
}

# 记录构建
record_build() {
    local version_tag=$1
    local full_image=$2
    local build_date=$(date -Iseconds)
    local git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    local image_size=$(docker images "${full_image}" --format "{{.Size}}" 2>/dev/null || echo "unknown")

    local record_entry="## ${version_tag}

- **构建日期**: ${build_date}
- **Git 提交**: \`${git_commit}\`
- **镜像地址**: \`${full_image}\`
- **镜像大小**: ${image_size}
- **平台**: ${PLATFORM}

---"

    # 插入到记录文件开头
    if [ -f "${BUILD_RECORD}" ]; then
        local temp_file=$(mktemp)
        head -n 5 "${BUILD_RECORD}" > "${temp_file}"
        echo "" >> "${temp_file}"
        echo "${record_entry}" >> "${temp_file}"
        tail -n +6 "${BUILD_RECORD}" >> "${temp_file}"
        mv "${temp_file}" "${BUILD_RECORD}"
    else
        cat > "${BUILD_RECORD}" <<EOF
# SenseVoice 镜像构建记录

> 本记录跟踪所有镜像构建和推送历史

${record_entry}
EOF
    fi

    log_success "构建记录已更新: ${BUILD_RECORD}"
}

# 登录镜像仓库
login_registry() {
    log_info "请登录 SiliconFlow 镜像仓库..."
    docker login "${REGISTRY}" || {
        log_error "登录失败，请检查凭据"
        exit 1
    }
    log_success "登录成功"
}

# 显示使用帮助
show_help() {
    echo "Usage: $0 [OPTIONS] [VERSION_TAG]

Options:
    -h, --help       显示帮助信息
    -v, --version    显示版本号
    --no-login       跳过登录直接构建
    --rtx5090        构建 RTX 5090 (Blackwell) 版本

Arguments:
    VERSION_TAG      版本标签 (默认: 自动生成，如 20250123-v1.0.0)

Examples:
    $0                     # 自动生成版本标签 (RTX 4090 版本)
    $0 v1.0.0              # 使用指定标签 (RTX 4090 版本)
    $0 --rtx5090           # 构建 RTX 5090 版本
    $0 --rtx5090 v1.0.0    # 构建指定版本的 RTX 5090 版本
    $0 --no-login          # 跳过登录
"
}

# 主函数
main() {
    local version_tag=""
    local skip_login=false
    local target_arch="rtx4090"  # 默认构建 RTX 4090 版本

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                echo "SenseVoice Build Script v1.0.0"
                exit 0
                ;;
            --no-login)
                skip_login=true
                shift
                ;;
            --rtx5090)
                target_arch="rtx5090"
                shift
                ;;
            -*)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
            *)
                version_tag=$1
                shift
                ;;
        esac
    done

    echo "========================================"
    echo "  SenseVoice Docker 构建推送工具"
    echo "========================================"
    echo ""

    # 生成版本标签
    if [ -z "${version_tag}" ]; then
        version_tag=$(get_version_tag)
        log_info "自动生成版本标签: ${version_tag}"
    else
        log_info "使用指定版本标签: ${version_tag}"
    fi

    # 检查环境
    check_buildx
    setup_builder

    # 登录
    if [ "${skip_login}" = false ]; then
        login_registry
    else
        log_warn "跳过登录步骤"
    fi

    # 构建并推送
    local full_image=$(build_image "${version_tag}" "${target_arch}")
    if [ $? -eq 0 ]; then
        record_build "${version_tag}" "${full_image}"
        echo ""
        echo "========================================"
        log_success "镜像构建推送完成!"
        echo "镜像地址: ${full_image}"
        echo "========================================"
    else
        log_error "镜像构建推送失败"
        exit 1
    fi
}

# 运行主函数
main "$@"