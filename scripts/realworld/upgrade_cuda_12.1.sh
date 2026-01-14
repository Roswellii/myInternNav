#!/bin/bash
# CUDA 12.1 升级脚本
# 注意：此脚本需要 root 权限或 sudo 权限

set -e  # 遇到错误立即退出

echo "=========================================="
echo "CUDA 12.1 升级脚本"
echo "=========================================="

# 检查是否为 root 或使用 sudo
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  此脚本需要 root 权限，请使用 sudo 运行"
    echo "   使用方法: sudo bash upgrade_cuda_12.1.sh"
    exit 1
fi

# 检查当前 CUDA 版本
echo ""
echo "1. 检查当前 CUDA 版本..."
if command -v nvcc &> /dev/null; then
    CURRENT_CUDA=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c2-)
    echo "   当前 CUDA 版本: $CURRENT_CUDA"
else
    echo "   CUDA 未安装或不在 PATH 中"
fi

# 检查 NVIDIA 驱动
echo ""
echo "2. 检查 NVIDIA 驱动..."
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -1)
    echo "   驱动版本: $DRIVER_VERSION"
    echo "   支持 CUDA 版本: $CUDA_VERSION"
    
    # 检查是否支持 CUDA 12.1
    if [[ $(echo "$CUDA_VERSION >= 12.1" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        echo "   ✅ 驱动支持 CUDA 12.1"
    else
        echo "   ⚠️  驱动可能不支持 CUDA 12.1，建议先升级驱动"
        read -p "   是否继续? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "   ⚠️  nvidia-smi 未找到，请确保 NVIDIA 驱动已安装"
    exit 1
fi

# 设置 CUDA 版本
CUDA_VERSION="12.1"
CUDA_PATCH="0"
CUDA_FULL_VERSION="${CUDA_VERSION}.${CUDA_PATCH}"

# 检测系统架构
ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
    ARCH="x86_64"
else
    echo "❌ 不支持的架构: $ARCH"
    exit 1
fi

# 检测操作系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    echo "❌ 无法检测操作系统"
    exit 1
fi

echo ""
echo "3. 系统信息:"
echo "   操作系统: $OS $VER"
echo "   架构: $ARCH"
echo "   目标 CUDA 版本: $CUDA_FULL_VERSION"

# 选择安装方式
echo ""
echo "4. 选择安装方式:"
echo "   1) 使用 apt 安装 (推荐，适用于 Ubuntu/Debian)"
echo "   2) 使用 runfile 安装 (适用于所有 Linux 发行版)"
read -p "   请选择 (1/2): " INSTALL_METHOD

if [ "$INSTALL_METHOD" == "1" ]; then
    # 使用 apt 安装
    echo ""
    echo "5. 使用 apt 安装 CUDA 12.1..."
    
    # 添加 NVIDIA 仓库
    if [ "$OS" == "ubuntu" ] || [ "$OS" == "debian" ]; then
        # 下载并添加密钥
        wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}${VER//./}/${ARCH}/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update
        
        # 安装 CUDA 12.1
        apt-get install -y cuda-toolkit-12-1
        
        # 设置环境变量
        echo ""
        echo "6. 设置环境变量..."
        cat >> /etc/profile.d/cuda.sh << 'EOF'
# CUDA 12.1
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
        
        # 创建符号链接
        if [ -L /usr/local/cuda ]; then
            rm /usr/local/cuda
        fi
        ln -s /usr/local/cuda-12.1 /usr/local/cuda
        
        echo "   ✅ CUDA 12.1 安装完成"
    else
        echo "   ❌ apt 安装方式仅支持 Ubuntu/Debian"
        echo "   请使用 runfile 安装方式"
        exit 1
    fi
    
elif [ "$INSTALL_METHOD" == "2" ]; then
    # 使用 runfile 安装
    echo ""
    echo "5. 使用 runfile 安装 CUDA 12.1..."
    
    # 下载 CUDA 12.1
    CUDA_RUNFILE="cuda_${CUDA_VERSION}.${CUDA_PATCH}_${ARCH}.run"
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_FULL_VERSION}/local_installers/${CUDA_RUNFILE}"
    
    echo "   下载 CUDA 12.1 runfile..."
    cd /tmp
    wget "$CUDA_URL" -O "$CUDA_RUNFILE"
    
    # 运行安装程序
    echo "   运行安装程序..."
    echo "   注意：安装过程中会询问是否安装驱动，建议选择 'no'（如果已安装驱动）"
    sh "$CUDA_RUNFILE" --toolkit --silent --override
    
    # 设置环境变量
    echo ""
    echo "6. 设置环境变量..."
    cat >> /etc/profile.d/cuda.sh << 'EOF'
# CUDA 12.1
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
    
    # 创建符号链接
    if [ -L /usr/local/cuda ]; then
        rm /usr/local/cuda
    fi
    ln -s /usr/local/cuda-12.1 /usr/local/cuda
    
    echo "   ✅ CUDA 12.1 安装完成"
    
    # 清理
    rm -f /tmp/$CUDA_RUNFILE
else
    echo "❌ 无效的选择"
    exit 1
fi

# 验证安装
echo ""
echo "7. 验证安装..."
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

if command -v nvcc &> /dev/null; then
    NEW_CUDA=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c2-)
    echo "   ✅ CUDA 版本: $NEW_CUDA"
else
    echo "   ⚠️  nvcc 未找到，可能需要重新加载环境变量"
    echo "   请运行: source /etc/profile.d/cuda.sh"
fi

echo ""
echo "=========================================="
echo "CUDA 12.1 升级完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 重新加载环境变量:"
echo "   source /etc/profile.d/cuda.sh"
echo "   或重新登录终端"
echo ""
echo "2. 验证安装:"
echo "   nvcc --version"
echo ""
echo "3. 重新安装 PyTorch (支持 CUDA 12.1):"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "4. 验证 PyTorch CUDA:"
echo "   python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda)'"
echo ""














