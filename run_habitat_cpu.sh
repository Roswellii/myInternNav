#!/bin/bash
# 使用CPU渲染运行habitat评估

echo "=== 使用CPU渲染运行Habitat评估 ==="

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# 设置环境变量以使用CPU渲染
export HABITAT_SIM_HEADLESS=1  # 启用headless模式
unset EGL_VISIBLE_DEVICES  # 清除EGL设备设置

# 设置配置文件路径
CONFIG_FILE="${1:-scripts/eval/configs/habitat_dual_system_cfg.py}"

echo "配置文件: $CONFIG_FILE"
echo "HABITAT_SIM_HEADLESS: $HABITAT_SIM_HEADLESS"
echo "EGL_VISIBLE_DEVICES: ${EGL_VISIBLE_DEVICES:-未设置}"
echo ""

# 运行评估
python scripts/eval/eval.py --config "$CONFIG_FILE"













