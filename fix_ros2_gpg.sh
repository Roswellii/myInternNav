#!/bin/bash

# 修复 ROS2 GPG 密钥问题

set -e

echo "=========================================="
echo "修复 ROS2 GPG 密钥"
echo "=========================================="

# 1. 删除有问题的源文件
echo ""
echo "步骤 1: 删除有问题的源文件..."
sudo rm -f /etc/apt/sources.list.d/ros2-latest.list

# 2. 确保 keyrings 目录存在
echo ""
echo "步骤 2: 创建 keyrings 目录..."
sudo mkdir -p /usr/share/keyrings

# 3. 下载并添加 GPG 密钥
echo ""
echo "步骤 3: 下载 ROS2 GPG 密钥..."
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -o /usr/share/keyrings/ros-archive-keyring.gpg

# 4. 验证密钥文件
if [ -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
    echo "GPG 密钥文件已成功创建"
    ls -lh /usr/share/keyrings/ros-archive-keyring.gpg
else
    echo "错误: GPG 密钥文件创建失败"
    exit 1
fi

# 5. 重新添加源（针对 Ubuntu 22.04，使用 Focal 源）
echo ""
echo "步骤 4: 重新添加 ROS2 Foxy 源..."
OS_VERSION=$(lsb_release -rs)
if [ "$OS_VERSION" = "22.04" ]; then
    echo "检测到 Ubuntu 22.04，使用 Ubuntu 20.04 (Focal) 的 ROS2 Foxy 源"
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/ros2-latest.list
else
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2-latest.list
fi

# 6. 更新 apt
echo ""
echo "步骤 5: 更新 apt 包列表..."
sudo apt update

echo ""
echo "=========================================="
echo "GPG 密钥修复完成！"
echo "=========================================="
echo ""
echo "如果还有问题，可以尝试以下备用方法："
echo "  sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654"
echo ""





















