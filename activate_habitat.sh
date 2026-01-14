#!/bin/bash
# 激活habitat环境并检查GPU可用性

echo "=== 激活Habitat环境 ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

echo ""
echo "=== 环境信息 ==="
echo "Python版本: $(python --version)"
echo "Python路径: $(which python)"
echo ""

echo "=== 检查Habitat安装 ==="
python << 'EOF'
try:
    import habitat
    print(f"✓ habitat: {getattr(habitat, '__version__', 'installed')}")
except ImportError as e:
    print(f"✗ habitat未安装: {e}")

try:
    import habitat_sim
    print(f"✓ habitat_sim: {getattr(habitat_sim, '__version__', 'installed')}")
except ImportError as e:
    print(f"✗ habitat_sim未安装: {e}")
EOF

echo ""
echo "=== GPU状态 ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU: %s\n驱动版本: %s\n显存: %s MiB / %s MiB\n利用率: %s%%\n", $1, $2, $4, $3, $5}'

echo ""
echo "=== PyTorch CUDA支持 ==="
python << 'EOF'
try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("  ✗ CUDA不可用")
except ImportError as e:
    print(f"✗ PyTorch未安装: {e}")
EOF

echo ""
echo "=== 环境已激活 ==="
echo "您现在可以使用habitat环境了！"
echo ""
echo "提示: 要运行habitat评估，可以使用:"
echo "  python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py"
echo ""
echo "要退出环境，输入: conda deactivate"





