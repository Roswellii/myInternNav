#!/bin/bash
# 使用GPU渲染运行habitat评估（带可视化）

echo "=== 使用GPU渲染运行Habitat评估 ==="

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# 设置EGL环境变量以使用NVIDIA驱动（覆盖conda的Mesa配置）
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export EGL_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# 设置库路径优先使用系统的NVIDIA库
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# 设置配置文件路径
CONFIG_FILE="${1:-scripts/eval/configs/habitat_dual_system_cfg.py}"

echo "配置文件: $CONFIG_FILE"
echo "__EGL_VENDOR_LIBRARY_DIRS: $__EGL_VENDOR_LIBRARY_DIRS"
echo "__EGL_VENDOR_LIBRARY_FILENAMES: $__EGL_VENDOR_LIBRARY_FILENAMES"
echo ""

# 运行评估
python scripts/eval/eval.py --config "$CONFIG_FILE"

