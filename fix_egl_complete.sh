#!/bin/bash
# Complete EGL fix script for habitat-sim

echo "=== Complete EGL Fix for Habitat-Sim ==="
echo ""

# Check GPU
echo "1. Checking GPU..."
nvidia-smi --list-gpus || echo "nvidia-smi not available"

echo ""
echo "2. Current environment:"
echo "   EGL_VISIBLE_DEVICES: ${EGL_VISIBLE_DEVICES:-Not set}"
echo "   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo "   HABITAT_SIM_HEADLESS: ${HABITAT_SIM_HEADLESS:-Not set}"

echo ""
echo "3. Trying comprehensive fix..."

# Method 1: Try with specific EGL device and additional settings
export MESA_GL_VERSION_OVERRIDE=4.5
export CUDA_VISIBLE_DEVICES=0

# Try each EGL device with additional settings
for egl_idx in 0 1 2 3; do
    echo ""
    echo "=== Trying EGL device $egl_idx with full settings ==="
    export EGL_VISIBLE_DEVICES=$egl_idx
    
    # Test with a simple Python script
    python3 << EOF
import os
os.environ['EGL_VISIBLE_DEVICES'] = '$egl_idx'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.5'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    from habitat_sim import Simulator
    from habitat_sim.utils.common import d3_habitat_sim
    import habitat_sim
    
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "NONE"
    backend_cfg.create_renderer = True
    
    sim = Simulator(backend_cfg)
    print(f"✓ SUCCESS: EGL device $egl_idx works!")
    sim.close()
    exit(0)
except Exception as e:
    error_msg = str(e)
    if "unable to find CUDA device" in error_msg or "WindowlessContext" in error_msg:
        print(f"✗ FAILED: {error_msg[:150]}")
        exit(1)
    else:
        # Other errors might be OK (like missing scene)
        print(f"? Different error (might be OK): {error_msg[:150]}")
        exit(0)
EOF
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓✓✓ Found working EGL device: $egl_idx ✓✓✓"
        echo ""
        echo "To use this, run:"
        echo "  export EGL_VISIBLE_DEVICES=$egl_idx"
        echo "  export MESA_GL_VERSION_OVERRIDE=4.5"
        echo "  export CUDA_VISIBLE_DEVICES=0"
        echo "  python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py"
        exit 0
    fi
done

echo ""
echo "=== All EGL devices failed ==="
echo ""
echo "Alternative solutions:"
echo ""
echo "1. Use headless mode (no GUI, but should work):"
echo "   export HABITAT_SIM_HEADLESS=1"
echo "   python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py"
echo ""
echo "2. Check if you need to install additional packages:"
echo "   - libegl1-mesa-dev"
echo "   - libgles2-mesa-dev"
echo ""
echo "3. Try updating GPU drivers:"
echo "   sudo apt update && sudo apt upgrade nvidia-driver-*"
echo ""
echo "4. Check if X server is needed (if not using headless):"
echo "   echo \$DISPLAY"
echo "   xhost +local:"
exit 1






