#!/bin/bash
# Fix Vulkan GPU access in Docker container

echo "=== Vulkan GPU Access Diagnosis ==="
echo ""

echo "1. Checking NVIDIA libraries:"
docker exec internnav bash -c "ldconfig -p | grep -i 'libnvidia-compute' | head -3" || echo "  ✗ NVIDIA compute libraries not found"

echo ""
echo "2. Checking Vulkan ICD files:"
docker exec internnav bash -c "ls -la /usr/share/vulkan/icd.d/ 2>&1" | grep -v "^total" || echo "  ✗ Vulkan ICD directory not found"

echo ""
echo "3. Checking for NVIDIA Vulkan driver:"
docker exec internnav bash -c "find /usr -name '*nvidia*icd*.json' 2>/dev/null | head -3" || echo "  ✗ NVIDIA Vulkan ICD file not found"

echo ""
echo "4. Current environment variables:"
docker exec internnav bash -c "env | grep -E '(VK|CUDA|NVIDIA)' | sort"

echo ""
echo "=== Solutions ==="
echo ""
echo "Problem: Vulkan cannot find NVIDIA GPU"
echo ""
echo "Solution 1: Ensure container was started with --gpus all"
echo "  Check: docker inspect internnav | grep -i gpu"
echo ""
echo "Solution 2: Set VK_ICD_FILENAMES environment variable"
echo "  docker exec internnav bash -c 'export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json'"
echo ""
echo "Solution 3: Use headless mode (no Vulkan needed)"
echo "  Set headless=True in config file"
echo ""
echo "Solution 4: Check if nvidia-container-toolkit is installed on host"
echo "  Check: nvidia-container-toolkit --version"
echo ""
echo "⚠️  If NVIDIA Vulkan ICD is missing from container, you may need to:"
echo "  1. Use a Docker image with NVIDIA Vulkan drivers pre-installed"
echo "  2. Install NVIDIA Vulkan drivers inside the container (complex)"
echo "  3. Use headless mode instead (recommended for servers)"















