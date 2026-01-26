#!/bin/bash
# 降级 transformers 到 4.49.0

set -e

echo "=========================================="
echo "Transformers 降级到 4.49.0"
echo "=========================================="

# 检查 conda 环境
if command -v conda &> /dev/null; then
    CONDA_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
    echo "当前 conda 环境: $CONDA_ENV"
    
    if [ -n "$CONDA_ENV" ] && [ "$CONDA_ENV" != "base" ]; then
        echo "激活环境: $CONDA_ENV"
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
    fi
else
    echo "⚠️  conda 未找到，将使用系统 Python"
fi

# 检查当前版本
echo ""
echo "1. 检查当前 transformers 版本..."
if python3 -c "import transformers" 2>/dev/null; then
    CURRENT_VERSION=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
    echo "   当前版本: $CURRENT_VERSION"
else
    echo "   transformers 未安装"
fi

# 降级
echo ""
echo "2. 降级 transformers 到 4.49.0..."
pip install transformers==4.49.0

# 验证
echo ""
echo "3. 验证安装..."
python3 -c "
import transformers
print(f'✅ Transformers 版本: {transformers.__version__}')
if transformers.__version__.startswith('4.49'):
    print('✅ 降级成功！')
else:
    print(f'⚠️  版本不匹配，当前版本: {transformers.__version__}')
"

echo ""
echo "=========================================="
echo "降级完成！"
echo "=========================================="
echo ""
echo "注意："
echo "1. 如果遇到依赖冲突，可能需要重新安装相关依赖"
echo "2. 建议测试模型是否正常工作"
echo "3. 如果出现问题，可以回退：pip install transformers==4.51.0"
echo ""






















