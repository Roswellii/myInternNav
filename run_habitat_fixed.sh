#!/bin/bash
# 使用LD_PRELOAD强制使用NVIDIA EGL库运行habitat

echo "=== 使用LD_PRELOAD修复Habitat EGL问题 ==="

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# 设置所有环境变量
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export EGL_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# 使用LD_PRELOAD强制使用系统的NVIDIA EGL库
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0:${LD_PRELOAD}

# 设置配置文件路径
CONFIG_FILE="${1:-scripts/eval/configs/habitat_dual_system_cfg.py}"

echo "配置文件: $CONFIG_FILE"
echo "LD_PRELOAD: $LD_PRELOAD"
echo ""

# 运行评估
python scripts/eval/eval.py --config "$CONFIG_FILE"













