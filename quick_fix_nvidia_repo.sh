#!/bin/bash
# 快速替换NVIDIA源为国内镜像

set -e

if [ "$EUID" -ne 0 ]; then 
    echo "请使用sudo运行此脚本: sudo bash $0"
    exit 1
fi

echo "替换nvidia-container-toolkit为中科大镜像..."
sed -i 's|https://nvidia.github.io|https://mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
echo "✓ 已完成"

echo ""
echo "优化apt下载参数..."
cat > /etc/apt/apt.conf.d/99download-optimization << 'EOF'
# 优化下载性能
Acquire::http::Max-Age "3600";
Acquire::http::Pipeline-Depth "10";
Acquire::Queue-Mode "access";
Acquire::http::Timeout "600";
Acquire::ftp::Timeout "600";
Acquire::ForceIPv4 "true";
Acquire::Retries "10";
EOF
echo "✓ 已完成"

echo ""
echo "当前配置："
echo "NVIDIA Container Toolkit源："
cat /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo ""
echo "注意：NVIDIA CUDA仓库(developer.download.nvidia.com)没有镜像源，"
echo "如果下载仍然慢，建议配置代理或使用VPN"














