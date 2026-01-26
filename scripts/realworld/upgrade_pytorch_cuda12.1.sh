#!/bin/bash
# 升级 PyTorch 以支持 CUDA 12.1

set -e

echo "=========================================="
echo "PyTorch CUDA 12.1 升级脚本"
echo "=========================================="

# 检查 CUDA 12.1 是否已安装
echo "1. 检查 CUDA 12.1 安装..."
if [ -d "/usr/local/cuda-12.1" ]; then
    echo "   ✅ CUDA 12.1 已安装"
    export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
else
    echo "   ⚠️  CUDA 12.1 未找到，请先运行 upgrade_cuda_12.1.sh"
    read -p "   是否继续安装 PyTorch? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 conda 环境
echo ""
echo "2. 检查 conda 环境..."
if command -v conda &> /dev/null; then
    CONDA_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
    echo "   当前 conda 环境: $CONDA_ENV"
    
    if [ -n "$CONDA_ENV" ] && [ "$CONDA_ENV" != "base" ]; then
        echo "   激活环境: $CONDA_ENV"
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
    fi
else
    echo "   ⚠️  conda 未找到，将使用系统 Python"
fi

# 检查当前 PyTorch 版本
echo ""
echo "3. 检查当前 PyTorch 版本..."
if python3 -c "import torch" 2>/dev/null; then
    CURRENT_PYTORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    CURRENT_CUDA=$(python3 -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null)
    echo "   当前 PyTorch: $CURRENT_PYTORCH"
    echo "   当前 CUDA (PyTorch): $CURRENT_CUDA"
else
    echo "   PyTorch 未安装"
fi

# 卸载旧版本
echo ""
echo "4. 卸载旧版本 PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# 安装新版本
echo ""
echo "5. 安装 PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证安装
echo ""
echo "6. 验证安装..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA 不可用')
"

echo ""
echo "=========================================="
echo "PyTorch CUDA 12.1 升级完成！"
echo "=========================================="






















