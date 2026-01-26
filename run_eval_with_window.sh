#!/bin/bash
# Run evaluation with proper environment variables for window display

echo "=== Starting Evaluation with Window Support ==="
echo ""

# Set all required environment variables
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export EGL_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=${DISPLAY:-:1}
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export QT_X11_NO_MITSHM=1
export MESA_GL_VERSION_OVERRIDE=4.5

echo "Environment variables set:"
echo "  VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo "  EGL_VISIBLE_DEVICES=$EGL_VISIBLE_DEVICES"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  DISPLAY=$DISPLAY"
echo ""

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container"
    CONFIG_FILE="${1:-scripts/eval/configs/h1_internvla_n1_async_cfg.py}"
    python scripts/eval/eval.py --config "$CONFIG_FILE"
else
    echo "Running on host - use this inside Docker container:"
    echo "  docker exec -it internnav bash -c 'source /root/InternNav/run_eval_with_window.sh'"
    echo ""
    echo "Or copy these environment variables and run manually:"
    env | grep -E '(VK|EGL|CUDA|DISPLAY|ACCEPT|PRIVACY|QT|MESA)' | while read line; do
        echo "  export $line"
    done
fi
















