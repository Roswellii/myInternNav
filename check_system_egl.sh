#!/bin/bash
# Check system configuration for EGL support

echo "=== System EGL Configuration Check ==="
echo ""

echo "1. Checking GPU and drivers:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

echo ""
echo "2. Checking EGL libraries:"
ldconfig -p | grep -i egl | head -5 || echo "  No EGL libraries found in ldconfig"

echo ""
echo "3. Checking installed EGL packages:"
dpkg -l | grep -i "libegl\|libgles" | head -10 || echo "  No EGL packages found (or not using dpkg)"

echo ""
echo "4. Checking environment variables:"
echo "   DISPLAY: ${DISPLAY:-Not set}"
echo "   EGL_VISIBLE_DEVICES: ${EGL_VISIBLE_DEVICES:-Not set}"
echo "   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"

echo ""
echo "5. Testing EGL with Python:"
python3 << 'EOF'
import sys
try:
    from OpenGL import EGL
    print("  ✓ PyOpenGL with EGL available")
except ImportError:
    print("  ✗ PyOpenGL with EGL not available")
    print("    Try: pip install PyOpenGL PyOpenGL-accelerate")

try:
    import os
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.5'
    # Try to import habitat-sim to see if it can detect EGL
    from habitat_sim.utils.common import d3_habitat_sim
    print("  ✓ habitat-sim can import EGL utilities")
except ImportError as e:
    print(f"  ✗ habitat-sim EGL import failed: {e}")
except Exception as e:
    print(f"  ? habitat-sim EGL check: {e}")
EOF

echo ""
echo "=== Recommendations ==="
echo ""
echo "If EGL is not working, try:"
echo ""
echo "1. Install EGL libraries:"
echo "   sudo apt-get update"
echo "   sudo apt-get install libegl1-mesa-dev libgles2-mesa-dev"
echo ""
echo "2. Check if you need X server (for non-headless):"
echo "   echo \$DISPLAY"
echo "   # If empty, you may need: export DISPLAY=:0"
echo ""
echo "3. Try the comprehensive fix script:"
echo "   ./fix_egl_complete.sh"
echo ""
echo "4. If all else fails, use headless mode (no GUI but functional):"
echo "   export HABITAT_SIM_HEADLESS=1"
echo "   python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py"















