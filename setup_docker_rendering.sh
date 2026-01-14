#!/bin/bash
# Setup Docker rendering environment variables
# Run this inside the Docker container

echo "=== Setting up Docker Rendering Environment ==="

# Essential Isaac Sim variables
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

# Display configuration (try common values)
if [ -z "$DISPLAY" ]; then
    # Try to detect DISPLAY from host
    if [ -n "$HOST_DISPLAY" ]; then
        export DISPLAY=$HOST_DISPLAY
    else
        export DISPLAY=:0
    fi
    echo "Set DISPLAY=$DISPLAY"
else
    echo "DISPLAY already set to: $DISPLAY"
fi

# GPU configuration
if [ -z "$EGL_VISIBLE_DEVICES" ]; then
    export EGL_VISIBLE_DEVICES=0
    echo "Set EGL_VISIBLE_DEVICES=0"
else
    echo "EGL_VISIBLE_DEVICES already set to: $EGL_VISIBLE_DEVICES"
fi

# Optional: Set CUDA device if needed
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES not set (will use all available GPUs)"
else
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# GL/Mesa configuration
export MESA_GL_VERSION_OVERRIDE=4.5
export QT_X11_NO_MITSHM=1

# Isaac Sim specific
export OMNI_USER_HOME=/root/.local/share/ov
export OMNIVERSECACHE=/root/.cache/ov

echo ""
echo "✓ Environment variables set:"
echo "  DISPLAY=$DISPLAY"
echo "  EGL_VISIBLE_DEVICES=$EGL_VISIBLE_DEVICES"
echo "  ACCEPT_EULA=$ACCEPT_EULA"
echo "  PRIVACY_CONSENT=$PRIVACY_CONSENT"
echo "  MESA_GL_VERSION_OVERRIDE=$MESA_GL_VERSION_OVERRIDE"
echo ""
echo "To make these permanent, add them to your ~/.bashrc or container entrypoint"








