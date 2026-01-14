#!/bin/bash
# 使用Xvfb虚拟显示运行habitat评估（带可视化）

echo "=== 使用Xvfb虚拟显示运行Habitat评估 ==="

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# 设置EGL环境变量
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export EGL_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# 设置配置文件路径
CONFIG_FILE="${1:-scripts/eval/configs/habitat_dual_system_cfg.py}"

echo "配置文件: $CONFIG_FILE"
echo ""

# 检查Xvfb是否可用
if ! command -v Xvfb &> /dev/null; then
    echo "错误: Xvfb未安装。请安装: sudo apt-get install xvfb"
    exit 1
fi

# 启动Xvfb虚拟显示
echo "启动Xvfb虚拟显示..."
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
export DISPLAY=:99

# 等待Xvfb启动
sleep 2

# 检查Xvfb是否运行
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "错误: Xvfb启动失败"
    exit 1
fi

echo "Xvfb运行中 (PID: $XVFB_PID, DISPLAY: $DISPLAY)"
echo ""

# 运行评估
echo "运行Habitat评估..."
python scripts/eval/eval.py --config "$CONFIG_FILE"
EXIT_CODE=$?

# 清理：停止Xvfb
echo ""
echo "停止Xvfb..."
kill $XVFB_PID 2>/dev/null
wait $XVFB_PID 2>/dev/null

exit $EXIT_CODE





