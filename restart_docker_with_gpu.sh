#!/bin/bash
# Restart Docker container with proper GPU configuration

echo "=== Restarting Docker Container with GPU Support ==="
echo ""

CONTAINER_NAME="internnav"

# Check if container is running
if docker ps --filter name=$CONTAINER_NAME --format "{{.Names}}" | grep -q $CONTAINER_NAME; then
    echo "Stopping container $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME
fi

# Remove container if exists
if docker ps -a --filter name=$CONTAINER_NAME --format "{{.Names}}" | grep -q $CONTAINER_NAME; then
    echo "Removing container $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME
fi

echo ""
echo "Container stopped and removed."
echo ""
echo "⚠️  Now you need to restart the container with proper GPU configuration:"
echo ""
echo "docker run --name $CONTAINER_NAME -it --rm \\"
echo "  --gpus all \\"
echo "  --runtime=nvidia \\"
echo "  --network host \\"
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
echo "Note: Add --runtime=nvidia flag to ensure nvidia-container-runtime is used"
echo "      If --runtime=nvidia doesn't work, try --gpus all (which should be enough)"















