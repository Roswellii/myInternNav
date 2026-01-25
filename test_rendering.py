#!/usr/bin/env python3
"""Test if rendering is working in Docker container"""

import os
import sys

print("=== Rendering Test ===")
print("")

# Check environment variables
print("1. Environment Variables:")
env_vars = [
    'DISPLAY', 'EGL_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
    'ACCEPT_EULA', 'PRIVACY_CONSENT', 'MESA_GL_VERSION_OVERRIDE'
]
for var in env_vars:
    value = os.environ.get(var, 'Not set')
    print(f"   {var}: {value}")

print("")
print("2. Testing GPU Access:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"   ✓ GPU available: {result.stdout.strip().split(chr(10))[0]}")
    else:
        print("   ✗ nvidia-smi failed")
except Exception as e:
    print(f"   ✗ GPU check failed: {e}")

print("")
print("3. Testing Isaac Sim/Omniverse:")
try:
    import omni
    print("   ✓ Isaac Sim packages importable")
    
    # Try to import common rendering modules
    try:
        from omni.isaac.core import World
        print("   ✓ Isaac Core importable")
    except Exception as e:
        print(f"   ? Isaac Core import: {e}")
        
except ImportError as e:
    print(f"   ✗ Isaac Sim not available: {e}")
except Exception as e:
    print(f"   ? Isaac Sim check: {e}")

print("")
print("4. Testing EGL/OpenGL:")
try:
    # Try to check if EGL devices are available
    # This might work even without PyOpenGL installed
    result = subprocess.run(['eglinfo'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✓ eglinfo works (EGL available)")
    else:
        print("   ? eglinfo not available (this is OK)")
except FileNotFoundError:
    print("   ? eglinfo not installed (this is OK, not required)")
except Exception as e:
    print(f"   ? EGL check: {e}")

print("")
print("5. Testing Actual Rendering:")
print("   Attempting to create a simple Isaac Sim world...")
try:
    from omni.isaac.core import World
    
    # Try to create a headless world (minimal rendering)
    world = World(stage_units_in_meters=1.0)
    print("   ✓ Isaac Sim World created successfully")
    print("   ✓ Rendering should work!")
    
    world.clear()
    print("   ✓ World cleared")
    
except Exception as e:
    error_msg = str(e)
    if "GLFW" in error_msg or "GPU Foundation" in error_msg or "Window" in error_msg:
        print(f"   ⚠ Warning: {error_msg[:100]}")
        print("   This is common in Docker. Try setting:")
        print("     export EGL_VISIBLE_DEVICES=0")
        print("   If your program works, you can ignore these warnings.")
    else:
        print(f"   ✗ Error: {error_msg[:100]}")

print("")
print("=== Summary ===")
print("")
print("If you see GLFW/GPU Foundation warnings but Isaac Sim can create a World,")
print("then rendering is working correctly. These warnings are from Isaac Sim")
print("plugins trying to initialize GUI windows, but EGL offscreen rendering")
print("should still work.")
print("")
print("To suppress warnings, ensure these are set:")
print("  export EGL_VISIBLE_DEVICES=0")
print("  export ACCEPT_EULA=Y")
print("  export PRIVACY_CONSENT=Y")















