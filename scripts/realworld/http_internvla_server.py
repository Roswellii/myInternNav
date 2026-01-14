#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
import threading
from queue import Queue, Empty

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''
first_request = True  # Flag to force reset on first request

# Action mapping for visualization
ACTION_NAMES = {
    0: 'STOP',
    1: 'FORWARD',
    2: 'LEFT',
    3: 'RIGHT',
    4: 'LOOK_UP',
    5: 'LOOK_DOWN',
}

# Visualization queue for thread-safe communication
vis_queue = Queue(maxsize=10)


def visualization_thread():
    """Dedicated thread for matplotlib visualization (runs in main thread context)."""
    plt.ion()  # Enable interactive mode
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title('InternNav Visualization')
    
    # Initialize right plot placeholder
    axes[1].text(
        0.5, 0.5, 'No Pixel Goal\n(Discrete Action Mode)',
        transform=axes[1].transAxes,
        fontsize=14, ha='center', va='center',
        color='#666666', style='italic'
    )
    axes[1].set_facecolor('#f0f0f0')
    axes[1].set_title('Pixel Goal', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    print("Visualization window initialized. Waiting for data...")
    
    while True:
        try:
            # Get data from queue (blocking with timeout)
            data = vis_queue.get(timeout=0.1)
            
            if data is None:  # Poison pill to stop thread
                break
            
            image = data['image']
            instruction = data['instruction']
            action = data.get('action')
            pixel_goal = data.get('pixel_goal')
            step_idx = data.get('step_idx', 0)
            
            # Clear left plot only
            axes[0].clear()
            
            # Left plot: Current image with action
            axes[0].imshow(image)
            axes[0].set_title(f'Step {step_idx} - Current View', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Add action text on the image
            if action is not None:
                if isinstance(action, list):
                    action_text = ', '.join([ACTION_NAMES.get(a, str(a)) for a in action])
                else:
                    action_text = ACTION_NAMES.get(action, str(action))
                axes[0].text(
                    0.5, 0.02, f'Action: {action_text}',
                    transform=axes[0].transAxes,
                    fontsize=14, fontweight='bold',
                    color='white', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#2E86AB', alpha=0.9)
                )
            
            # Right plot: Only update when pixel_goal exists
            if pixel_goal is not None:
                axes[1].clear()
                axes[1].imshow(image)
                axes[1].set_title('Pixel Goal', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                # Draw pixel goal marker (pixel_goal is [y, x])
                goal_y, goal_x = pixel_goal[0], pixel_goal[1]
                axes[1].scatter(goal_x, goal_y, c='#FF6B6B', s=300, marker='*', edgecolors='white', linewidths=2, zorder=5)
                axes[1].plot(goal_x, goal_y, 'o', markersize=20, markerfacecolor='none', markeredgecolor='#FF6B6B', markeredgewidth=3)
                axes[1].text(
                    goal_x + 15, goal_y - 15, f'({goal_x}, {goal_y})',
                    fontsize=11, fontweight='bold', color='#FF6B6B',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
                )
            
            # Add instruction as super title
            wrapped_instruction = '\n'.join([instruction[i:i+80] for i in range(0, len(instruction), 80)])
            fig.suptitle(f'Navigation Instruction:\n"{wrapped_instruction}"', fontsize=11, fontweight='bold', color='#2C3E50')
            
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

            # Save current visualization to disk so it is preserved even after closing the window
            try:
                from pathlib import Path
                # Prefer run-specific output_dir if available, otherwise fall back to a default logs path
                save_root = output_dir if output_dir else os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "logs",
                    "realworld_vis"
                )
                Path(save_root).mkdir(parents=True, exist_ok=True)
                save_path = os.path.join(save_root, f"vis_step_{step_idx:04d}.png")
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            except Exception as e:
                # Do not crash visualization due to save errors
                print(f"Failed to save visualization figure: {e}")
            
        except Empty:
            # Queue is empty, just continue waiting
            continue
        except Exception as e:
            import traceback
            print(f"Visualization error: {e}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            time.sleep(0.1)
            continue
    
    plt.close(fig)
    print("Visualization thread stopped.")


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time, first_request
    start_time = time.time()

    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}")

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    instruction = "move forward until past the desk and turn left until you see a two doors. Exit this office from the nearest door.Do not go to the far door.  after you exit, turn right. move forward then turn to the hall way on you left and stop immediatelty"
    policy_init = data['reset']
    
    # Force reset on first request after server start
    if first_request:
        policy_init = True
        first_request = False
        print("First request after server start, forcing reset!")
    
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    idx += 1

    look_down = False
    t0 = time.time()
    dual_sys_output = {}

    dual_sys_output = agent.step(
        image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
    )
    if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
        look_down = True
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )

    json_output = {}
    pixel_goal = None
    action = None
    
    if dual_sys_output.output_action is not None:
        json_output['discrete_action'] = dual_sys_output.output_action
        action = dual_sys_output.output_action
    else:
        json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
        if dual_sys_output.output_pixel is not None:
            json_output['pixel_goal'] = dual_sys_output.output_pixel
            pixel_goal = dual_sys_output.output_pixel

    # Send data to visualization thread
    try:
        vis_data = {
            'image': image.copy(),
            'instruction': instruction,
            'action': action,
            'pixel_goal': pixel_goal,
            'step_idx': idx
        }
        if not vis_queue.full():
            vis_queue.put(vis_data, block=False)
    except Exception as e:
        print(f"Failed to send data to visualization queue: {e}")

    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-wo-dagger")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=8)
    args = parser.parse_args()

    # Convert relative model_path to absolute path based on project root
    if not os.path.isabs(args.model_path):
        # Get project root directory (parent of internnav package)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.model_path = os.path.join(project_root, args.model_path)
    
    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    agent = InternVLAN1AsyncAgent(args)
    agent.reset()  # Create save_dir before first step
    agent.step(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640), dtype=np.float32),
        np.eye(4),
        "hello",
        intrinsic=args.camera_intrinsic,
    )
    agent.reset()  # Clear past_key_values cache after warmup

    # Start Flask in a separate thread so visualization can run in main thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host='172.20.10.6', port=5801, threaded=True, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    print(f"Flask server started on http://172.20.10.6:5801")
    
    # Run visualization in main thread (required for matplotlib GUI)
    try:
        visualization_thread()
    except KeyboardInterrupt:
        print("\nShutting down...")
        vis_queue.put(None)  # Send poison pill
