#!/bin/bash
# Script to install habitat dependencies and verify installation

echo "=== Installing habitat dependencies ==="

# Activate habitat conda environment
echo "Activating habitat conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate habitat

# Check if habitat-sim and habitat-lab are installed
echo ""
echo "Checking habitat installation..."
python3 -c "import habitat; print('habitat version:', habitat.__version__)" 2>&1 || echo "habitat not installed"
python3 -c "import habitat_baselines; print('habitat_baselines available')" 2>&1 || echo "habitat_baselines not installed"

# Install depth_camera_filtering if needed
echo ""
echo "Installing depth_camera_filtering..."
pip install git+https://github.com/naokiyokoyama/depth_camera_filtering.git || echo "Failed to install depth_camera_filtering"

# Install internnav with habitat extra
echo ""
echo "Installing internnav with habitat extra..."
pip install -e .[habitat]

# Test imports
echo ""
echo "Testing imports..."
python3 test_habitat_import.py

echo ""
echo "=== Done ==="















