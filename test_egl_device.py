#!/usr/bin/env python3
"""Test different EGL device indices to find the correct one"""

import os
import sys

def test_egl_device(egl_idx):
    """Test if a specific EGL device index works"""
    os.environ['EGL_VISIBLE_DEVICES'] = str(egl_idx)
    print(f"\nTesting EGL_VISIBLE_DEVICES={egl_idx}...")
    
    try:
        # Try to import and create a minimal habitat-sim context
        from habitat_sim import Simulator
        from habitat_sim.utils.common import d3_habitat_sim
        
        # Create a minimal config
        import habitat_sim
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = "NONE"  # We don't need a real scene for testing
        
        # Try to create simulator (this will fail if EGL device is wrong)
        # But we catch the error to test
        try:
            sim = Simulator(backend_cfg)
            sim.close()
            print(f"✓ EGL device {egl_idx} works!")
            return True
        except Exception as e:
            error_msg = str(e)
            if "unable to find CUDA device" in error_msg or "WindowlessContext" in error_msg:
                print(f"✗ EGL device {egl_idx} failed: {error_msg[:100]}")
                return False
            else:
                # Other errors might be OK (like missing scene)
                print(f"? EGL device {egl_idx} - different error (might be OK): {error_msg[:100]}")
                return True
    except ImportError as e:
        print(f"Could not import habitat_sim: {e}")
        return False
    except Exception as e:
        print(f"Error testing EGL device {egl_idx}: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing EGL Devices ===\n")
    print("This script will test different EGL device indices to find one that works.")
    print("If you see 'unable to find CUDA device' errors, try the next index.\n")
    
    working_devices = []
    for egl_idx in range(4):  # Test devices 0-3
        if test_egl_device(egl_idx):
            working_devices.append(egl_idx)
    
    print("\n=== Results ===")
    if working_devices:
        print(f"Working EGL device(s): {working_devices}")
        print(f"\nTo use, set: export EGL_VISIBLE_DEVICES={working_devices[0]}")
    else:
        print("No working EGL devices found in range 0-3.")
        print("You may need to:")
        print("1. Check your GPU drivers")
        print("2. Use headless mode: export HABITAT_SIM_HEADLESS=1")
        print("3. Try other EGL device indices")























