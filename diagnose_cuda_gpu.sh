#!/bin/bash
# Diagnose CUDA GPU access issue in Docker

echo "=== CUDA GPU Access Diagnosis ==="
echo ""

echo "1. Checking host GPU:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -1

echo ""
echo "2. Checking container GPU access:"
docker exec internnav bash -c "nvidia-smi --query-gpu=name --format=csv,noheader 2>&1" || echo "  ✗ nvidia-smi failed in container"

echo ""
echo "3. Checking NVIDIA device files in container:"
docker exec internnav bash -c "ls -la /dev/nvidia* 2>&1 | wc -l" && echo "  ✓ NVIDIA devices found" || echo "  ✗ No NVIDIA devices"

echo ""
echo "4. Checking container runtime:"
docker inspect internnav 2>/dev/null | grep -A 5 '"Runtime"' | head -5

echo ""
echo "5. Checking CUDA libraries in container:"
docker exec internnav bash -c "ldconfig -p | grep -i 'libcuda\|libnvidia-compute' | head -3" || echo "  ✗ CUDA libraries not found"

echo ""
echo "6. Environment variables in container:"
docker exec internnav bash -c "env | grep -E '(CUDA|NVIDIA|GPU)' | sort"

echo ""
echo "=== Problem Analysis ==="
echo ""
echo "Error: 'no CUDA-capable device is detected'"
echo ""
echo "Possible causes:"
echo "  1. Container not using nvidia-container-runtime"
echo "  2. NVIDIA driver mismatch between host and container"
echo "  3. CUDA_VISIBLE_DEVICES set incorrectly"
echo "  4. GPU context already initialized by another process"
echo ""
echo "=== Solutions ==="
echo ""
echo "Solution 1: Restart container with --gpus all"
echo "  docker stop internnav"
echo "  docker rm internnav"
echo "  # Then restart with --gpus all flag"
echo ""
echo "Solution 2: Check if container uses nvidia runtime"
echo "  docker inspect internnav | grep -i runtime"
echo "  # Should show: \"Runtime\": \"nvidia\""
echo ""
echo "Solution 3: Unset CUDA_VISIBLE_DEVICES if set incorrectly"
echo "  # The code sets it to local_rank (usually 0)"
echo "  # If local_rank is wrong, it may hide the GPU"
echo ""
echo "Solution 4: Check for GPU resource conflicts"
echo "  nvidia-smi  # Check if GPU is in use by other processes"








