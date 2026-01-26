#!/usr/bin/env python3
"""Diagnose EGL device issues for habitat-sim"""

import os
import subprocess
import sys

print("=== EGL Device Diagnosis ===\n")

# Check CUDA devices
print("1. Checking CUDA devices:")
try:
    result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
        print(f"Found {gpu_count} GPU(s)")
    else:
        print("nvidia-smi failed")
except Exception as e:
    print(f"Error running nvidia-smi: {e}")

print("\n2. Checking EGL devices:")
try:
    # Try to query EGL devices using habitat-sim utilities
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    from habitat_sim.utils.common import d3_habitat_sim
    
    # Try to create a simulator to see available devices
    print("Attempting to detect EGL devices...")
    # This might fail, but we'll catch it
except Exception as e:
    print(f"Could not query EGL devices directly: {e}")

print("\n3. Current environment variables:")
print(f"  EGL_VISIBLE_DEVICES: {os.environ.get('EGL_VISIBLE_DEVICES', 'Not set')}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"  HABITAT_SIM_HEADLESS: {os.environ.get('HABITAT_SIM_HEADLESS', 'Not set')}")

print("\n4. Testing different EGL device indices:")
print("   Try setting EGL_VISIBLE_DEVICES to different values (0, 1, 2, 3)")
print("   Example: EGL_VISIBLE_DEVICES=0 python scripts/eval/eval.py --config ...")
print("   Or: EGL_VISIBLE_DEVICES=1 python scripts/eval/eval.py --config ...")

print("\n5. Alternative solution:")
print("   If EGL devices don't match CUDA devices, you can:")
print("   - Use headless mode: export HABITAT_SIM_HEADLESS=1")
print("   - Or try different EGL device indices until one works")























