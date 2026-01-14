#!/bin/bash

# NVIDIA 535驱动安装脚本
# 注意：此脚本会卸载当前驱动并安装535版本，需要重启系统

set -e

echo "=========================================="
echo "NVIDIA 535 驱动安装脚本"
echo "=========================================="
echo ""

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then 
    echo "请使用sudo运行此脚本"
    exit 1
fi

# 显示当前驱动版本
echo "当前驱动版本："
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "未检测到NVIDIA驱动"

echo ""
echo "开始安装NVIDIA 535驱动..."
echo ""

# 1. 卸载当前NVIDIA驱动
echo "步骤1: 卸载当前NVIDIA驱动..."
apt-get remove --purge -y '^nvidia-.*' 2>/dev/null || true
apt-get remove --purge -y '^libnvidia-.*' 2>/dev/null || true
apt-get autoremove -y

# 2. 更新包列表
echo ""
echo "步骤2: 更新包列表..."
apt-get update

# 3. 安装NVIDIA 535驱动
echo ""
echo "步骤3: 安装NVIDIA 535驱动..."
apt-get install -y nvidia-driver-535

# 4. 安装完成后提示
echo ""
echo "=========================================="
echo "驱动安装完成！"
echo "=========================================="
echo ""
echo "请重启系统以使新驱动生效："
echo "  sudo reboot"
echo ""
echo "重启后，运行以下命令验证安装："
echo "  nvidia-smi"
echo ""







