#!/bin/bash

# 脚本用于移除系统中的 CUDA 11.8
# 需要 root 权限运行

set -e

echo "开始移除 CUDA 11.8..."

# 1. 卸载所有 CUDA 11.8 相关的 apt 包
echo "步骤 1: 卸载 CUDA 11.8 相关的 apt 包..."
PACKAGES=$(dpkg -l | grep -i cuda | grep 11.8 | awk '{print $2}')
if [ -n "$PACKAGES" ]; then
    echo "找到以下包将被移除:"
    echo "$PACKAGES"
    echo ""
    sudo apt-get remove --purge -y $PACKAGES
    echo "包卸载完成"
else
    echo "未找到 CUDA 11.8 相关的包"
fi

# 2. 清理 apt 缓存
echo ""
echo "步骤 2: 清理 apt 缓存..."
sudo apt-get autoremove -y
sudo apt-get autoclean

# 3. 删除符号链接
echo ""
echo "步骤 3: 删除符号链接..."
if [ -L /usr/local/cuda-11 ]; then
    echo "删除 /usr/local/cuda-11 符号链接..."
    sudo rm -f /usr/local/cuda-11
fi

if [ -L /etc/alternatives/cuda-11 ]; then
    echo "删除 /etc/alternatives/cuda-11 符号链接..."
    sudo rm -f /etc/alternatives/cuda-11
fi

# 4. 删除 CUDA 11.8 目录
echo ""
echo "步骤 4: 删除 /usr/local/cuda-11.8 目录..."
if [ -d /usr/local/cuda-11.8 ]; then
    echo "删除 /usr/local/cuda-11.8 目录..."
    sudo rm -rf /usr/local/cuda-11.8
    echo "目录删除完成"
else
    echo "/usr/local/cuda-11.8 目录不存在"
fi

# 5. 配置 CUDA 12.1 为默认版本
echo ""
echo "步骤 5: 配置 CUDA 12.1 为默认版本..."
if [ -d /usr/local/cuda-12.1 ]; then
    # 创建 /usr/local/cuda 符号链接指向 12.1
    if [ -L /usr/local/cuda ]; then
        echo "更新 /usr/local/cuda 符号链接指向 CUDA 12.1..."
        sudo rm -f /usr/local/cuda
    elif [ -e /usr/local/cuda ]; then
        echo "警告: /usr/local/cuda 已存在但不是符号链接，跳过创建"
    else
        echo "创建 /usr/local/cuda 符号链接指向 CUDA 12.1..."
    fi
    sudo ln -sf /usr/local/cuda-12.1 /usr/local/cuda
    echo "✓ /usr/local/cuda -> /usr/local/cuda-12.1"
    
    # 更新 .bashrc 中的环境变量
    echo ""
    echo "更新 ~/.bashrc 中的 CUDA 环境变量..."
    if [ -f ~/.bashrc ]; then
        # 备份 .bashrc
        cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
        echo "已备份 ~/.bashrc"
        
        # 替换 CUDA 11.8 为 12.1
        sed -i 's|/usr/local/cuda-11.8|/usr/local/cuda-12.1|g' ~/.bashrc
        
        # 如果不存在 CUDA 环境变量，则添加
        if ! grep -q "CUDA_HOME=/usr/local/cuda-12.1" ~/.bashrc; then
            echo "" >> ~/.bashrc
            echo "# CUDA 12.1 environment variables" >> ~/.bashrc
            echo "export PATH=/usr/local/cuda-12.1/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ~/.bashrc
            echo "export CUDA_HOME=/usr/local/cuda-12.1" >> ~/.bashrc
            echo "export CUDA_ROOT=/usr/local/cuda-12.1" >> ~/.bashrc
            echo "已添加 CUDA 12.1 环境变量到 ~/.bashrc"
        else
            echo "✓ ~/.bashrc 中的 CUDA 环境变量已更新为 12.1"
        fi
    else
        echo "~/.bashrc 不存在，创建新的配置文件..."
        cat >> ~/.bashrc << 'EOF'

# CUDA 12.1 environment variables
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_ROOT=/usr/local/cuda-12.1
EOF
        echo "已创建 ~/.bashrc 并添加 CUDA 12.1 环境变量"
    fi
    
    echo ""
    echo "✓ CUDA 12.1 配置完成"
    echo "提示: 请运行 'source ~/.bashrc' 或重新打开终端以使环境变量生效"
else
    echo "⚠ 警告: /usr/local/cuda-12.1 目录不存在，无法配置"
fi

# 6. 验证移除结果
echo ""
echo "步骤 6: 验证移除结果..."
echo "检查剩余的 CUDA 11.8 包:"
REMAINING=$(dpkg -l | grep -i cuda | grep 11.8 | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    echo "✓ 所有 CUDA 11.8 包已移除"
else
    echo "⚠ 仍有 $REMAINING 个 CUDA 11.8 相关的包"
    dpkg -l | grep -i cuda | grep 11.8
fi

echo ""
echo "检查 /usr/local/cuda-11.8 目录:"
if [ ! -d /usr/local/cuda-11.8 ]; then
    echo "✓ /usr/local/cuda-11.8 目录已删除"
else
    echo "⚠ /usr/local/cuda-11.8 目录仍然存在"
fi

echo ""
echo "=========================================="
echo "CUDA 11.8 移除完成！"
echo "=========================================="
echo ""
echo "当前系统中的 CUDA 版本:"
ls -la /usr/local/ | grep cuda || echo "未找到 CUDA 安装"
echo ""
echo "当前 CUDA 环境变量配置:"
if [ -L /usr/local/cuda ]; then
    echo "  /usr/local/cuda -> $(readlink -f /usr/local/cuda)"
fi
echo ""
echo "~/.bashrc 中的 CUDA 配置:"
grep -E "CUDA|/usr/local/cuda" ~/.bashrc | grep -v "^#" || echo "  未找到 CUDA 配置"
echo ""
echo "提示: 运行以下命令使环境变量生效:"
echo "  source ~/.bashrc"
echo "  或者重新打开终端"
echo ""
echo "验证 CUDA 12.1 是否可用:"
echo "  nvcc --version"
echo "  echo \$CUDA_HOME"

