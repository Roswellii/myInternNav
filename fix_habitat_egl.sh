#!/bin/bash
# Script to fix EGL/CUDA device mismatch for habitat-sim

echo "=== Fixing Habitat EGL/CUDA Device Issue ==="

# Check available EGL devices
echo ""
echo "Checking EGL devices..."
python3 << 'EOF'
import os
os.environ['HABITAT_SIM_LOG'] = 'quiet'
try:
    from habitat_sim.utils.common import d3_habitat_sim
    print("habitat-sim available")
except ImportError:
    print("habitat-sim not available")

# Try to list EGL devices
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\nAvailable GPUs:")
        print(result.stdout)
    else:
        print("nvidia-smi not available")
except:
    print("Could not check GPUs")
EOF

echo ""
echo "Setting environment variables for EGL..."
echo ""
echo "Option 1: Set EGL_VISIBLE_DEVICES to match CUDA device"
echo "  export EGL_VISIBLE_DEVICES=0"
echo ""
echo "Option 2: Use CUDA_VISIBLE_DEVICES to limit visible devices"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo ""
echo "Option 3: Set HABITAT_SIM_HEADLESS and use specific EGL device"
echo "  export HABITAT_SIM_HEADLESS=1"
echo "  export EGL_VISIBLE_DEVICES=0"
echo ""
echo "To run with fix, use:"
echo "  EGL_VISIBLE_DEVICES=0 python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py"
echo ""






















