#!/bin/bash
# Fix Docker rendering configuration for Isaac Sim/GPU rendering

echo "=== Docker Rendering Configuration Fix ==="
echo ""

# Check if running inside Docker
if [ -f /.dockerenv ]; then
    echo "✓ Running inside Docker container"
    IN_DOCKER=true
else
    echo "⚠ Running on host (not in Docker)"
    IN_DOCKER=false
fi

echo ""
echo "1. Checking X11/Display configuration:"

# Check DISPLAY
if [ -z "$DISPLAY" ]; then
    echo "  ✗ DISPLAY is not set"
    echo "    Try: export DISPLAY=:0"
    echo "    Or: export DISPLAY=:10.0"
else
    echo "  ✓ DISPLAY=$DISPLAY"
fi

# Check X11 socket
if [ -S /tmp/.X11-unix/X0 ] || [ -S /tmp/.X11-unix/X${DISPLAY##*:} ]; then
    echo "  ✓ X11 socket found"
else
    echo "  ✗ X11 socket not found"
    if [ "$IN_DOCKER" = true ]; then
        echo "    Note: This might be OK if X11 socket is mounted from host"
    else
        echo "    You may need to mount X11 socket in Docker:"
        echo "    -v /tmp/.X11-unix/:/tmp/.X11-unix"
    fi
fi

echo ""
echo "2. Checking GPU access:"

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "  ✓ nvidia-smi works"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
    else
        echo "  ✗ nvidia-smi failed (GPU not accessible)"
        echo "    Docker needs --gpus all flag"
    fi
else
    echo "  ✗ nvidia-smi not found"
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    echo "  ✓ CUDA compiler found"
else
    echo "  ? CUDA compiler not found (might be OK if using pre-built packages)"
fi

echo ""
echo "3. Checking EGL configuration:"

if [ -z "$EGL_VISIBLE_DEVICES" ]; then
    echo "  ? EGL_VISIBLE_DEVICES not set"
    echo "    Try: export EGL_VISIBLE_DEVICES=0"
else
    echo "  ✓ EGL_VISIBLE_DEVICES=$EGL_VISIBLE_DEVICES"
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "  ? CUDA_VISIBLE_DEVICES not set (using all GPUs)"
else
    echo "  ✓ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

echo ""
echo "4. Checking Isaac Sim/Omniverse environment variables:"

if [ -z "$ACCEPT_EULA" ]; then
    echo "  ? ACCEPT_EULA not set"
    echo "    Set: export ACCEPT_EULA=Y"
else
    echo "  ✓ ACCEPT_EULA=$ACCEPT_EULA"
fi

if [ -z "$PRIVACY_CONSENT" ]; then
    echo "  ? PRIVACY_CONSENT not set"
    echo "    Set: export PRIVACY_CONSENT=Y"
else
    echo "  ✓ PRIVACY_CONSENT=$PRIVACY_CONSENT"
fi

echo ""
echo "=== Recommendations ==="
echo ""

if [ "$IN_DOCKER" = false ]; then
    echo "📋 To run Docker container with rendering support, use:"
    echo ""
    echo "# Step 1: Allow X11 access"
    echo "xhost +local:root"
    echo ""
    echo "# Step 2: Run Docker with GPU and display support"
    echo "docker run --name internnav -it --rm --gpus all --network host \\"
    echo "  -e \"ACCEPT_EULA=Y\" \\"
    echo "  -e \"PRIVACY_CONSENT=Y\" \\"
    echo "  -e \"DISPLAY=\${DISPLAY}\" \\"
    echo "  --entrypoint /bin/bash \\"
    echo "  -w /root/InternNav \\"
    echo "  -v /tmp/.X11-unix/:/tmp/.X11-unix \\"
    echo "  -v \${PWD}:/root/InternNav \\"
    echo "  -v \${HOME}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \\"
    echo "  -v \${HOME}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \\"
    echo "  -v \${HOME}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \\"
    echo "  -v \${HOME}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \\"
    echo "  -v \${HOME}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \\"
    echo "  -v \${HOME}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \\"
    echo "  -v \${HOME}/docker/isaac-sim/data:/root/.local/share/ov/data:rw \\"
    echo "  -v \${HOME}/docker/isaac-sim/documents:/root/Documents:rw \\"
    echo "  your-docker-image:tag"
    echo ""
else
    echo "📋 If inside Docker, try setting these environment variables:"
    echo ""
    echo "export DISPLAY=\${DISPLAY:-:0}"
    echo "export EGL_VISIBLE_DEVICES=0"
    echo "export CUDA_VISIBLE_DEVICES=0"
    echo "export ACCEPT_EULA=Y"
    echo "export PRIVACY_CONSENT=Y"
    echo "export MESA_GL_VERSION_OVERRIDE=4.5"
    echo "export QT_X11_NO_MITSHM=1"
    echo ""
    echo "Note: If you still see GLFW/GPU Foundation errors but the program"
    echo "works, these are warnings and can be ignored if rendering works."
fi

echo ""
echo "=== Testing rendering ==="
echo ""

# Test EGL
python3 << 'EOF'
import os
import sys

print("Testing EGL...")
try:
    os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '4.5')
    # Try to import OpenGL
    from OpenGL import EGL
    print("  ✓ PyOpenGL EGL available")
except ImportError as e:
    print(f"  ✗ PyOpenGL EGL not available: {e}")
except Exception as e:
    print(f"  ? EGL check: {e}")

# Test if Isaac Sim can initialize
try:
    # This is just a check, might fail if Isaac Sim is not properly installed
    print("\nTesting Isaac Sim environment...")
    import subprocess
    result = subprocess.run(['python3', '-c', 'import omni'], 
                          capture_output=True, timeout=5)
    if result.returncode == 0:
        print("  ✓ Isaac Sim/Omniverse Python packages available")
    else:
        print("  ? Isaac Sim import failed (might be OK if not using Isaac Sim)")
except Exception as e:
    print(f"  ? Isaac Sim check: {e}")

EOF

echo ""
echo "=== Summary ==="
echo ""
echo "Common issues and fixes:"
echo ""
echo "1. 'Authorization required, but no authorization protocol specified'"
echo "   → Run: xhost +local:root (on host before starting Docker)"
echo ""
echo "2. 'GLFW initialization failed'"
echo "   → Ensure X11 is mounted: -v /tmp/.X11-unix/:/tmp/.X11-unix"
echo "   → Set DISPLAY: -e DISPLAY=\${DISPLAY}"
echo ""
echo "3. 'GPU Foundation is not initialized'"
echo "   → Ensure GPU is passed: --gpus all"
echo "   → Try setting: export EGL_VISIBLE_DEVICES=0"
echo ""
echo "4. If errors persist but program works:"
echo "   → These are warnings from Isaac Sim plugins"
echo "   → As long as rendering works, you can ignore them"
















