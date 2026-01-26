#!/bin/bash
# Script to run habitat evaluation with automatic EGL device detection

CONFIG_FILE="${1:-scripts/eval/configs/habitat_dual_system_cfg.py}"

echo "=== Running Habitat Evaluation ==="
echo "Config: $CONFIG_FILE"
echo ""

# Try different EGL device indices
for egl_idx in 0 1 2 3; do
    echo "Trying EGL_VISIBLE_DEVICES=$egl_idx..."
    export EGL_VISIBLE_DEVICES=$egl_idx
    
    # Run the evaluation
    python scripts/eval/eval.py --config "$CONFIG_FILE" 2>&1 | tee /tmp/habitat_eval_egl${egl_idx}.log
    
    # Check if it succeeded (no EGL error)
    if ! grep -q "unable to find CUDA device.*among.*EGL devices" /tmp/habitat_eval_egl${egl_idx}.log; then
        echo ""
        echo "✓ Success! EGL device $egl_idx works."
        echo "To use this device in the future, run:"
        echo "  export EGL_VISIBLE_DEVICES=$egl_idx"
        exit 0
    fi
    
    echo "✗ EGL device $egl_idx failed, trying next..."
    echo ""
done

echo "All EGL devices failed. Please check:"
echo "1. GPU drivers are installed correctly"
echo "2. Try running: python3 test_egl_device.py"
echo "3. Or use headless mode: export HABITAT_SIM_HEADLESS=1"
exit 1























