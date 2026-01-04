#!/bin/bash

# ROS2 Foxy 安装脚本
# 适用于 Ubuntu 20.04/22.04

set -e

echo "=========================================="
echo "开始安装 ROS2 Foxy"
echo "=========================================="

# 检查系统版本
OS_VERSION=$(lsb_release -rs)
echo "检测到系统版本: Ubuntu $OS_VERSION"

# 1. 设置 locale
echo ""
echo "步骤 1: 设置 locale..."
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 2. 添加 ROS2 apt 源
echo ""
echo "步骤 2: 添加 ROS2 apt 源..."

# 检查是否已添加源
if ! grep -q "packages.ros.org/ros2" /etc/apt/sources.list.d/ros2-latest.list 2>/dev/null; then
    sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    
    # 添加 ROS2 Foxy 源 (使用 Ubuntu 20.04 的源，因为 Foxy 只官方支持 Focal)
    sudo apt install -y curl gnupg lsb-release
    
    # 确保 keyrings 目录存在
    sudo mkdir -p /usr/share/keyrings
    
    # 使用新的方法添加 GPG 密钥（兼容 Ubuntu 22.04）
    echo "下载 ROS2 GPG 密钥..."
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -o /usr/share/keyrings/ros-archive-keyring.gpg
    
    # 验证密钥文件是否存在
    if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
        echo "警告: GPG 密钥文件创建失败，尝试备用方法..."
        # 备用方法：使用 apt-key（虽然已弃用，但在某些情况下仍然有效）
        sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - || {
            echo "尝试从 keyserver 添加密钥..."
            sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654 || true
        }
        USE_SIGNED_BY=""
    else
        USE_SIGNED_BY="signed-by=/usr/share/keyrings/ros-archive-keyring.gpg"
        echo "GPG 密钥文件创建成功"
    fi
    
    # 对于 Ubuntu 22.04，我们需要手动添加 Focal 的源
    if [ "$OS_VERSION" = "22.04" ]; then
        echo "检测到 Ubuntu 22.04，将使用 Ubuntu 20.04 (Focal) 的 ROS2 Foxy 源"
        if [ -n "$USE_SIGNED_BY" ]; then
            echo "deb [arch=$(dpkg --print-architecture) $USE_SIGNED_BY] http://packages.ros.org/ros2/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/ros2-latest.list
        else
            echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/ros2-latest.list
        fi
    else
        if [ -n "$USE_SIGNED_BY" ]; then
            echo "deb [arch=$(dpkg --print-architecture) $USE_SIGNED_BY] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2-latest.list
        else
            echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2-latest.list
        fi
    fi
else
    echo "ROS2 源已存在，跳过添加"
fi

# 3. 更新 apt
echo ""
echo "步骤 3: 更新 apt 包列表..."
sudo apt update

# 4. 安装 ROS2 Foxy Desktop
echo ""
echo "步骤 4: 安装 ROS2 Foxy Desktop (这可能需要一些时间)..."
sudo apt install -y ros-foxy-desktop

# 5. 安装开发工具
echo ""
echo "步骤 5: 安装开发工具..."
sudo apt install -y python3-argcomplete python3-colcon-common-extensions

# 6. 设置环境变量
echo ""
echo "步骤 6: 设置环境变量..."

# 检查 .bashrc 中是否已添加
if ! grep -q "source /opt/ros/foxy/setup.bash" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# ROS2 Foxy" >> ~/.bashrc
    echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
    echo "已添加到 ~/.bashrc"
else
    echo "环境变量已存在于 ~/.bashrc"
fi

# 7. 安装依赖
echo ""
echo "步骤 7: 安装额外的依赖..."
sudo apt install -y \
    python3-pip \
    python3-rosdep \
    python3-vcstool

# 初始化 rosdep
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    echo ""
    echo "初始化 rosdep..."
    sudo rosdep init || echo "rosdep 可能已初始化"
    rosdep update || echo "rosdep update 可能已执行过"
fi

echo ""
echo "=========================================="
echo "ROS2 Foxy 安装完成！"
echo "=========================================="
echo ""
echo "请执行以下命令来使环境变量生效："
echo "  source ~/.bashrc"
echo ""
echo "或者直接运行："
echo "  source /opt/ros/foxy/setup.bash"
echo ""
echo "验证安装，运行："
echo "  ros2 --help"
echo ""

