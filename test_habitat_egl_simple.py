#!/usr/bin/env python3
"""
Simple test script to find working EGL device for Habitat-sim.
Each test runs in a separate process to avoid crashes affecting other tests.
"""

import os
import sys
import subprocess
import tempfile

def create_test_script(egl_idx, config_path):
    """Create a temporary test script for a specific EGL device"""
    script_content = f'''#!/usr/bin/env python3
import os
import sys

# Set EGL device
os.environ['EGL_VISIBLE_DEVICES'] = '{egl_idx}'

# Remove headless mode for visualization
if 'HABITAT_SIM_HEADLESS' in os.environ:
    del os.environ['HABITAT_SIM_HEADLESS']

try:
    from habitat_baselines.config.default import get_config as get_habitat_config
    from habitat import Env
    import habitat.config
    
    config = get_habitat_config('{config_path}')
    
    # Set gpu_device_id
    with habitat.config.read_write(config):
        if hasattr(config.habitat.simulator, 'habitat_sim_v0'):
            if hasattr(config.habitat.simulator.habitat_sim_v0, 'gpu_device_id'):
                config.habitat.simulator.habitat_sim_v0.gpu_device_id = {egl_idx}
    
    # Try to create environment
    env = Env(config)
    
    # Test reset
    obs = env.reset()
    if obs is None:
        print("RESULT:FAIL:Reset returned None")
        sys.exit(1)
    
    # Test step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    env.close()
    print("RESULT:SUCCESS")
    sys.exit(0)
    
except SystemExit as e:
    sys.exit(e.code if e.code is not None else 1)
except Exception as e:
    error_msg = ""
    try:
        error_msg = str(e)
    except:
        error_msg = "Unknown error"
    
    try:
        if "unable to find CUDA device" in error_msg or "WindowlessContext" in error_msg:
            print("RESULT:FAIL:EGL_MISMATCH")
        else:
            # Safely truncate error message
            safe_msg = error_msg[:100] if len(error_msg) > 100 else error_msg
            print(f"RESULT:FAIL:{safe_msg}")
    except:
        print("RESULT:FAIL:Error formatting failed")
    sys.exit(1)
'''
    return script_content

def test_egl_device(egl_idx, config_path):
    """Test a specific EGL device index using subprocess"""
    print(f"\n{'='*60}")
    print(f"Testing EGL_VISIBLE_DEVICES={egl_idx}")
    print(f"{'='*60}")
    
    # Create temporary test script
    script_content = create_test_script(egl_idx, config_path)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        os.chmod(script_path, 0o755)
        
        # Run test in subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            env=os.environ.copy()
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        # Check for EGL errors in stderr
        if "unable to find CUDA device" in stderr or "WindowlessContext" in stderr:
            print(f"  ✗ EGL/CUDA device mismatch detected")
            print(f"  Error: {stderr.strip()[:200]}")
            return False
        
        # Check result
        if result.returncode == 0:
            if "RESULT:SUCCESS" in stdout:
                print(f"  ✓ Environment created successfully!")
                print(f"  ✓ Reset and step tests passed!")
                print(f"\n{'='*60}")
                print(f"✓ SUCCESS: EGL device {egl_idx} works!")
                print(f"{'='*60}\n")
                return True
            else:
                print(f"  ✗ Unexpected output: {stdout[:200]}")
                return False
        else:
            if "RESULT:FAIL:EGL_MISMATCH" in stdout:
                print(f"  ✗ EGL/CUDA device mismatch")
            elif "RESULT:FAIL" in stdout:
                print(f"  ✗ Failed: {stdout.split('RESULT:FAIL:')[1][:150] if 'RESULT:FAIL:' in stdout else 'Unknown error'}")
            else:
                print(f"  ✗ Failed with return code {result.returncode}")
                if stderr:
                    print(f"  Stderr: {stderr.strip()[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Test timed out after 30 seconds")
        return False
    except Exception as e:
        error_msg = str(e) if e else "Unknown error"
        print(f"  ✗ Error running test: {error_msg}")
        return False
    finally:
        # Clean up
        try:
            os.unlink(script_path)
        except:
            pass

def main():
    print("="*60)
    print("Habitat-sim EGL Device Test Script (Subprocess Version)")
    print("Testing different EGL device indices for visualization")
    print("="*60)
    
    # Check if config path is provided
    config_path = 'scripts/eval/configs/vln_r2r.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Usage: python test_habitat_egl_simple.py [config_path]")
        sys.exit(1)
    
    print(f"\nUsing config: {config_path}")
    print(f"Current environment variables:")
    print(f"  EGL_VISIBLE_DEVICES: {os.environ.get('EGL_VISIBLE_DEVICES', 'Not set')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  HABITAT_SIM_HEADLESS: {os.environ.get('HABITAT_SIM_HEADLESS', 'Not set')}")
    
    # Test EGL devices 0-3
    working_devices = []
    
    for egl_idx in range(4):
        try:
            if test_egl_device(egl_idx, config_path):
                working_devices.append(egl_idx)
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"\n  ✗ Unexpected error testing device {egl_idx}: {e}")
            continue
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    if working_devices:
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
        print("   python test_habitat_egl_simple.py")
        print("3. Try other EGL device indices (4, 5, etc.)")
        print("4. Check EGL installation:")
        print("   python -c 'from OpenGL import EGL; print(EGL)'")
        print("5. If visualization is not required, use headless mode:")
        print("   export HABITAT_SIM_HEADLESS=1")
    
    print("="*60)

if __name__ == "__main__":
    main()

