#!/bin/bash
# 使用中国镜像下载 InternVLA-N1-wo-dagger 模型

set -e

# 设置镜像环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 模型信息
MODEL_ID="InternRobotics/InternVLA-N1-wo-dagger"
SAVE_DIR="checkpoints/InternVLA-N1-wo-dagger"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "下载 InternVLA-N1-wo-dagger 模型"
echo "使用镜像: $HF_ENDPOINT"
echo "保存路径: $PROJECT_ROOT/$SAVE_DIR"
echo "=========================================="
echo ""

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 检查是否安装了 huggingface_hub
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "正在安装 huggingface_hub..."
    python3 -m pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple || {
        echo "安装失败，请手动运行: pip install huggingface_hub"
        exit 1
    }
fi

# 运行 Python 下载脚本
python3 "$SCRIPT_DIR/download_internvla_n1_wo_dagger.py"

echo ""
echo "✅ 下载完成！"




