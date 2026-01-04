#!/bin/bash
# Run habitat evaluation with automatic fallback to headless mode if EGL fails

CONFIG_FILE="${1:-scripts/eval/configs/habitat_dual_system_cfg.py}"

echo "=== Running Habitat Evaluation with EGL Fallback ==="
echo "Config: $CONFIG_FILE"
echo ""

# Set additional environment variables that might help
export MESA_GL_VERSION_OVERRIDE=4.5
export CUDA_VISIBLE_DEVICES=0

# Try with graphics first (if not explicitly set to headless)
if [ -z "$HABITAT_SIM_HEADLESS" ]; then
    echo "Attempting to run with graphics (EGL)..."
    
    # Try EGL device 0 first (most common)
    export EGL_VISIBLE_DEVICES=0
    
    # Run and capture output
    python scripts/eval/eval.py --config "$CONFIG_FILE" 2>&1 | tee /tmp/habitat_eval.log
    
    # Check if EGL error occurred
    if grep -q "unable to find CUDA device.*among.*EGL devices" /tmp/habitat_eval.log || \
       grep -q "WindowlessContext.*Unable to create" /tmp/habitat_eval.log; then
        echo ""
        echo "⚠ EGL graphics mode failed. Falling back to headless mode..."
        echo ""
        
        # Try headless mode
        export HABITAT_SIM_HEADLESS=1
        unset EGL_VISIBLE_DEVICES  # Clear EGL setting for headless
        
        echo "Running in headless mode (no GUI, but should work)..."
        python scripts/eval/eval.py --config "$CONFIG_FILE"
    else
        echo ""
        echo "✓ Successfully ran with graphics!"
    fi
else
    # Headless mode explicitly requested
    echo "Running in headless mode (HABITAT_SIM_HEADLESS is set)..."
    unset EGL_VISIBLE_DEVICES
    python scripts/eval/eval.py --config "$CONFIG_FILE"
fi






