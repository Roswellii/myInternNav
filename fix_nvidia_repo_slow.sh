#!/bin/bash
# 加速NVIDIA CUDA仓库下载的脚本
# NVIDIA CUDA仓库没有官方镜像，此脚本优化下载设置

set -e

echo "=========================================="
echo "NVIDIA CUDA 仓库加速优化脚本"
echo "=========================================="
echo ""

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then 
    echo "请使用sudo运行此脚本"
    exit 1
fi

# 1. 优化nvidia-container-toolkit使用中科大镜像
echo "步骤1: 配置nvidia-container-toolkit使用中科大镜像"
if [ -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
    if grep -q "nvidia.github.io" /etc/apt/sources.list.d/nvidia-container-toolkit.list; then
        sed -i 's|https://nvidia.github.io|https://mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
        echo "✓ nvidia-container-toolkit源已替换为中科大镜像"
    else
        echo "✓ nvidia-container-toolkit已使用镜像源"
    fi
else
    echo "⚠ 未找到nvidia-container-toolkit源配置"
fi
echo ""

# 2. 优化apt下载参数
echo "步骤2: 优化apt下载参数"
cat > /etc/apt/apt.conf.d/99download-optimization << 'EOF'
# 优化下载性能
Acquire::http::Max-Age "3600";
Acquire::http::Pipeline-Depth "10";
Acquire::Queue-Mode "access";

# 增加超时时间（从默认的5分钟增加到10分钟）
Acquire::http::Timeout "600";
Acquire::ftp::Timeout "600";

# 强制使用IPv4（避免IPv6连接慢的问题）
Acquire::ForceIPv4 "true";

# 增加重试次数
Acquire::Retries "10";
EOF
echo "✓ apt下载优化已配置"
echo ""

# 3. 显示当前源配置
echo "=========================================="
echo "当前源配置:"
echo ""
echo "CUDA源:"
cat /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list 2>/dev/null || echo "未找到"
echo ""
if [ -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
    echo "NVIDIA Container Toolkit源:"
    cat /etc/apt/sources.list.d/nvidia-container-toolkit.list
    echo ""
fi

echo "=========================================="
echo "优化完成！"
echo ""
echo "注意：NVIDIA CUDA官方仓库(developer.download.nvidia.com)在国内没有镜像源"
echo "如果下载仍然很慢，建议："
echo "1. 配置HTTP代理（编辑 /etc/apt/apt.conf.d/95proxies）"
echo "2. 在网络较好的时间段重试"
echo "3. 或者使用VPN/代理工具"
echo ""
echo "现在可以运行: sudo apt-get update && sudo apt-get install -y <package>"
echo "=========================================="

