#!/usr/bin/env python3
"""
Test script to find working EGL device for Habitat-sim with visualization support.
This script will try different EGL device indices until finding one that works.
"""

import os
import sys
import contextlib
import io

def test_habitat_env_with_egl(egl_idx, config_path='scripts/eval/configs/vln_r2r.yaml'):
    """Test if a specific EGL device index works with Habitat environment"""
    print(f"\n{'='*60}")
    print(f"Testing EGL_VISIBLE_DEVICES={egl_idx}")
    print(f"{'='*60}")
    
    # Set EGL device
    os.environ['EGL_VISIBLE_DEVICES'] = str(egl_idx)
    
    # Ensure we're not in headless mode (for visualization)
    if 'HABITAT_SIM_HEADLESS' in os.environ:
        del os.environ['HABITAT_SIM_HEADLESS']
    
    try:
        from habitat_baselines.config.default import get_config as get_habitat_config
        from habitat.config.default import get_agent_config
        from habitat import Env
        
        print(f"[Step 1] Loading Habitat config from: {config_path}")
        config = get_habitat_config(config_path)
        
        print(f"[Step 2] Setting gpu_device_id to {egl_idx}")
        import habitat.config
        with habitat.config.read_write(config):
            if hasattr(config.habitat.simulator, 'habitat_sim_v0'):
                if hasattr(config.habitat.simulator.habitat_sim_v0, 'gpu_device_id'):
                    config.habitat.simulator.habitat_sim_v0.gpu_device_id = egl_idx
                    print(f"  ✓ Updated gpu_device_id to {egl_idx}")
        
        print(f"[Step 3] Creating Habitat environment...")
        print(f"  EGL_VISIBLE_DEVICES={os.environ.get('EGL_VISIBLE_DEVICES')}")
        print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        # Capture stderr to detect EGL errors
        stderr_capture = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr_capture):
                # Try to create environment
                env = Env(config)
        except Exception as e:
            stderr_output = stderr_capture.getvalue()
            error_msg = str(e)
            
            # Check stderr for EGL errors
            if "unable to find CUDA device" in stderr_output or "WindowlessContext" in stderr_output:
                print(f"  ✗ EGL/CUDA device mismatch detected")
                print(f"  Error: {stderr_output.strip()[:150]}")
                return False
            
            # Re-raise if it's a different error
            raise
        
        # Check stderr for any EGL warnings/errors
        stderr_output = stderr_capture.getvalue()
        if stderr_output and ("unable to find CUDA device" in stderr_output or "WindowlessContext" in stderr_output):
            print(f"  ✗ EGL/CUDA device mismatch detected in stderr")
            print(f"  Error: {stderr_output.strip()[:150]}")
            if hasattr(env, 'close'):
                try:
                    env.close()
                except:
                    pass
            return False
        
        print(f"  ✓ Environment created successfully!")
        
        print(f"[Step 4] Testing reset...")
        observations = env.reset()
        
        if observations is not None:
            print(f"  ✓ Reset successful!")
            print(f"  Observation keys: {list(observations.keys())}")
            
            # Check if we have RGB observations (for visualization)
            if 'rgb' in observations:
                rgb_shape = observations['rgb'].shape
                print(f"  ✓ RGB observation shape: {rgb_shape}")
            
            print(f"[Step 5] Testing step...")
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"  ✓ Step successful!")
            
            print(f"\n{'='*60}")
            print(f"✓ SUCCESS: EGL device {egl_idx} works!")
            print(f"{'='*60}\n")
            
            env.close()
            return True
        else:
            print(f"  ✗ Reset returned None")
            env.close()
            return False
            
    except SystemExit:
        # Habitat-sim might call sys.exit() on error
        stderr_output = stderr_capture.getvalue() if 'stderr_capture' in locals() else ""
        if "unable to find CUDA device" in stderr_output or "WindowlessContext" in stderr_output:
            print(f"  ✗ EGL/CUDA device mismatch (SystemExit)")
            return False
        raise
    except Exception as e:
        error_msg = str(e)
        stderr_output = stderr_capture.getvalue() if 'stderr_capture' in locals() else ""
        
        # Check both exception message and stderr
        if "unable to find CUDA device" in error_msg or "WindowlessContext" in error_msg or \
           "unable to find CUDA device" in stderr_output or "WindowlessContext" in stderr_output:
            print(f"  ✗ EGL/CUDA device mismatch error")
            print(f"  Exception: {error_msg[:150]}")
            if stderr_output:
                print(f"  Stderr: {stderr_output.strip()[:150]}")
            return False
        elif "No module named" in error_msg:
            print(f"  → Missing dependency: {error_msg}")
            return None  # None means dependency issue, not EGL issue
        else:
            print(f"  ✗ Failed: {error_msg[:200]}")
            if stderr_output:
                print(f"  Stderr: {stderr_output.strip()[:150]}")
        
        return False

def main():
    print("="*60)
    print("Habitat-sim EGL Device Test Script")
    print("Testing different EGL device indices for visualization")
    print("="*60)
    
    # Check if config path is provided
    config_path = 'scripts/eval/configs/vln_r2r.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Usage: python test_habitat_egl_visualization.py [config_path]")
        sys.exit(1)
    
    print(f"\nUsing config: {config_path}")
    print(f"Current environment variables:")
    print(f"  EGL_VISIBLE_DEVICES: {os.environ.get('EGL_VISIBLE_DEVICES', 'Not set')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  HABITAT_SIM_HEADLESS: {os.environ.get('HABITAT_SIM_HEADLESS', 'Not set')}")
    
    # Save original EGL_VISIBLE_DEVICES if set
    original_egl = os.environ.get('EGL_VISIBLE_DEVICES')
    
    # Test EGL devices 0-3
    working_devices = []
    dependency_issue = False
    
    for egl_idx in range(4):
        try:
            result = test_habitat_env_with_egl(egl_idx, config_path)
            
            if result is None:
                # Dependency issue, stop testing
                dependency_issue = True
                break
            elif result:
                working_devices.append(egl_idx)
                # Found a working device, can stop or continue to find all
                # For now, let's continue to find all working devices
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except SystemExit as e:
            # Habitat-sim might call sys.exit() on error
            print(f"\n  ✗ SystemExit occurred (likely EGL error)")
            # Continue to next device
            continue
        except Exception as e:
            print(f"\n  ✗ Unexpected error: {e}")
            # Continue to next device
            continue
        finally:
            # Restore original EGL_VISIBLE_DEVICES for next iteration
            if original_egl is not None:
                os.environ['EGL_VISIBLE_DEVICES'] = original_egl
            elif 'EGL_VISIBLE_DEVICES' in os.environ:
                # Clear it if it wasn't originally set
                pass  # Keep it for next test
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    if dependency_issue:
        print("✗ Dependency issue detected. Please install required packages.")
        print("  Try: pip install habitat-sim habitat-lab")
    elif working_devices:
        print(f"✓ Found {len(working_devices)} working EGL device(s): {working_devices}")
        print(f"\nRecommended: Use EGL_VISIBLE_DEVICES={working_devices[0]}")
        print(f"\nTo use in your evaluation, run:")
        print(f"  export EGL_VISIBLE_DEVICES={working_devices[0]}")
        print(f"  python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py")
        print(f"\nOr add to your code:")
        print(f"  os.environ['EGL_VISIBLE_DEVICES'] = '{working_devices[0]}'")
    else:
        print("✗ No working EGL devices found in range 0-3")
        print("\nPossible solutions:")
        print("1. Check GPU drivers: nvidia-smi")
        print("2. Try setting CUDA_VISIBLE_DEVICES:")
        print("   export CUDA_VISIBLE_DEVICES=0")
        print("3. Try other EGL device indices (4, 5, etc.)")
        print("4. Check if you need to use headless mode:")
        print("   export HABITAT_SIM_HEADLESS=1")
        print("   (Note: This disables visualization)")
        print("5. Check EGL installation:")
        print("   python -c 'from OpenGL import EGL; print(EGL)'")
    
    print("="*60)

if __name__ == "__main__":
    main()

