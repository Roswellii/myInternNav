import argparse
import json
import os
import sys

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict

import numpy as np
import quaternion
import torch
import tqdm
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.image_utils import to_numpy_array
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for display
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, Exception):
    MATPLOTLIB_AVAILABLE = False

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import (
    chunk_token,
    open_image,
    split_and_clean,
    traj_to_actions,
)

# Try to import habitat modules - delay import to allow registration even if habitat is not installed
try:
    import habitat
    from depth_camera_filtering import filter_depth
    from habitat.config.default import get_agent_config
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from habitat.utils.visualizations.utils import images_to_video, observations_to_image
    from habitat_baselines.config.default import get_config as get_habitat_config
    # Import for Habitat registry side effects — do not remove
    import internnav.habitat_extensions.measures  # noqa: F401 # isort: skip
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    # Import for Habitat registry side effects — do not remove
    try:
        import internnav.habitat_extensions.measures  # noqa: F401 # isort: skip
    except ImportError:
        pass

DEFAULT_IMAGE_TOKEN = "<image>"


@Evaluator.register('habitat_vln')
class HabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        # Check if habitat is available
        if not HABITAT_AVAILABLE:
            raise RuntimeError(
                "Habitat modules are not available. "
                "Please install habitat-sim and habitat-lab to use HabitatVLNEvaluator. "
                "You can install them following the instructions at https://github.com/facebookresearch/habitat-sim"
            )
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=False,  # Disable fog of war to always show full map and goal
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
        # Enable renderer for visualization
        with habitat.config.read_write(self.config):
            if hasattr(self.config.habitat.simulator, 'habitat_sim_v0'):
                if hasattr(self.config.habitat.simulator.habitat_sim_v0, 'create_renderer'):
                    self.config.habitat.simulator.habitat_sim_v0.create_renderer = True
                if hasattr(self.config.habitat.simulator.habitat_sim_v0, 'gpu_device_id'):
                    # Use GPU for rendering if available
                    self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
            
            # Allow custom camera height override from eval_settings
            camera_height_override = cfg.eval_settings.get('camera_height', None)
            if camera_height_override is not None:
                # Modify camera sensor positions
                main_agent = self.config.habitat.simulator.agents.main_agent
                if hasattr(main_agent, 'sim_sensors'):
                    if 'rgb_sensor' in main_agent.sim_sensors:
                        if not hasattr(main_agent.sim_sensors.rgb_sensor, 'position') or main_agent.sim_sensors.rgb_sensor.position is None:
                            main_agent.sim_sensors.rgb_sensor.position = [0.0, camera_height_override, 0.0]
                        else:
                            main_agent.sim_sensors.rgb_sensor.position[1] = camera_height_override
                    if 'depth_sensor' in main_agent.sim_sensors:
                        if not hasattr(main_agent.sim_sensors.depth_sensor, 'position') or main_agent.sim_sensors.depth_sensor.position is None:
                            main_agent.sim_sensors.depth_sensor.position = [0.0, camera_height_override, 0.0]
                        else:
                            main_agent.sim_sensors.depth_sensor.position[1] = camera_height_override
                print(f"Camera height set to: {camera_height_override}m")
        
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        super().__init__(cfg, init_agent=False)

        # ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:{self.local_rank}")
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map={"": device},
            )
        elif self.model_args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor

        # refactor: this part used in three places
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.objectnav_instructions = ["Search for the {target_object}."]

        self.num_frames = self.model_args.num_frames
        self.num_future_steps = self.model_args.num_future_steps
        self.num_history = self.model_args.num_history

        # Load demo.jpg image for reference
        demo_image_path = "/home/zhangwenqi/workspace/myInternNav/logs/habitat/test_dual_system_0125_error4/demo.jpg"
        print(f"[evaluator_init] Attempting to load demo image from: {demo_image_path}")
        if os.path.exists(demo_image_path):
            self.demo_image = Image.open(demo_image_path).convert('RGB')
            print(f"[evaluator_init] ✓ Successfully loaded demo reference image from {demo_image_path}, size={self.demo_image.size}")
        else:
            print(f"[evaluator_init] ✗ Warning: demo.jpg not found at {demo_image_path}")
            self.demo_image = None

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        sucs, spls, oss, nes, _ = self._run_local_eval()

        return {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
        }

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        global_metrics["sucs"] etc. are global 1-D CPU tensors with all episodes.
        """
        sucs_all = global_metrics["sucs"]
        spls_all = global_metrics["spls"]
        oss_all = global_metrics["oss"]
        nes_all = global_metrics["nes"]

        # avoid /0 if no episodes
        denom = max(len(sucs_all), 1)

        return {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

    def _run_local_eval(self) -> None:  # noqa: C901
        """
        Run local evaluation on this rank.

        Important: if resuming from previous results, need to read from / write to "self.output_path/progress.json".
                    For each episode, save the result dict in jsonl format to that file.
                    In Env, the episodes are already filtered by this file, tasks that have the same (scene_id, episode_id) are skipped.


        Returns
        -------
        dict[str, Tensor]:
            {
                "sucs": [N_local],
                "spls": [N_local],
                "oss":  [N_local],
                "nes":  [N_local],
            }
        """
        # Create / get env
        # self.env = self.env  # HabitatEnv from DistributedEvaluator

        sucs, spls, oss, nes = [], [], [], []
        self.model.eval()

        # resume from previous results
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    if "scene_id" not in res:
                        print("This evaluation has already finished!")
                        return (
                            torch.tensor(sucs).to(self.device),
                            torch.tensor(spls).to(self.device),
                            torch.tensor(oss).to(self.device),
                            torch.tensor(nes).to(self.device),
                            torch.tensor(len(sucs)).to(self.device),
                        )
                    if self.rank == 0:  # noqa: F405
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        nes.append(res['ne'])

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")
        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = (
                episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
            )
            print("episode start", episode_instruction)

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False)

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            # Display initial frame in visualization window
            self.use_cv2 = False
            self.use_matplotlib = False
            
            print("Checking visualization options...")
            if CV2_AVAILABLE:
                try:
                    # Test if cv2.imshow actually works
                    test_img = np.zeros((10, 10, 3), dtype=np.uint8)
                    cv2.imshow('test', test_img)
                    cv2.destroyAllWindows()
                    self.use_cv2 = True
                    print("OpenCV display test passed")
                except Exception as e:
                    self.use_cv2 = False
                    print(f"OpenCV display test failed: {e}")
            
            if not self.use_cv2 and MATPLOTLIB_AVAILABLE:
                self.use_matplotlib = True
                print("Using Matplotlib for visualization")
            elif not self.use_cv2 and not MATPLOTLIB_AVAILABLE:
                print("Warning: Neither OpenCV nor Matplotlib available for visualization")
            
            # Get initial info for top-down map
            initial_info = self.env.get_metrics()
            initial_top_down_map = initial_info.get('top_down_map', None)
            # Handle case where top_down_map might be a dict or None
            if initial_top_down_map is not None and isinstance(initial_top_down_map, dict):
                # If it's a dict, try to get the actual map array
                initial_top_down_map = initial_top_down_map.get('map', None) if isinstance(initial_top_down_map, dict) else initial_top_down_map
            # Ensure it's a numpy array if not None
            if initial_top_down_map is not None and not isinstance(initial_top_down_map, np.ndarray):
                try:
                    initial_top_down_map = np.array(initial_top_down_map)
                except Exception:
                    initial_top_down_map = None
            
            if self.use_cv2:
                try:
                    rgb = observations['rgb']
                    vis_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    
                    # Add text overlay with episode info and instruction
                    y_offset = 30
                    cv2.putText(vis_img, f'Episode: {scene_id}_{episode_id:04d}', (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 35
                    cv2.putText(vis_img, 'Initializing...', (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Display full instruction (with word wrapping, no truncation)
                    y_offset += 35
                    instruction_text = episode_instruction  # Show full instruction
                    words = instruction_text.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        if len(test_line) > 60:  # Max characters per line
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                        else:
                            current_line = test_line
                    if current_line:
                        lines.append(current_line)
                    
                    # Display all lines (no limit)
                    for i, line in enumerate(lines):
                        if y_offset + i * 25 < rgb.shape[0] - 20:  # Don't overflow image
                            cv2.putText(vis_img, line, (10, y_offset + i * 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Add top-down map if available (keep full size, don't shrink too much)
                    if initial_top_down_map is not None and isinstance(initial_top_down_map, np.ndarray):
                        try:
                            # Keep map at reasonable size but don't shrink too much
                            map_height = rgb.shape[0]
                            if len(initial_top_down_map.shape) >= 2:
                                # Preserve aspect ratio, but ensure map is visible
                                aspect_ratio = initial_top_down_map.shape[1] / initial_top_down_map.shape[0]
                                map_width = int(map_height * aspect_ratio)
                                # Limit max width to avoid too wide display
                                max_map_width = rgb.shape[1] * 1.5  # 1.5x of RGB width
                                if map_width > max_map_width:
                                    map_width = int(max_map_width)
                                    map_height = int(map_width / aspect_ratio)
                                top_down_resized = cv2.resize(initial_top_down_map, (map_width, map_height))
                                if len(top_down_resized.shape) == 2:
                                    top_down_resized = cv2.cvtColor(top_down_resized, cv2.COLOR_GRAY2BGR)
                                # Pad to match RGB height if needed
                                if top_down_resized.shape[0] < rgb.shape[0]:
                                    pad_height = rgb.shape[0] - top_down_resized.shape[0]
                                    top_down_resized = cv2.copyMakeBorder(top_down_resized, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                                vis_img = np.hstack([vis_img, top_down_resized])
                        except Exception as e:
                            print(f"Warning: Failed to process initial top-down map: {e}")
                    
                    cv2.imshow('Habitat Navigation', vis_img)
                    cv2.waitKey(1)
                    print("Visualization window opened (OpenCV)")
                except Exception as e:
                    self.use_cv2 = False
                    print(f"OpenCV display failed: {e}")
            
            if self.use_matplotlib:
                try:
                    plt.ion()  # Turn on interactive mode
                    self.fig = plt.figure('Habitat Navigation', figsize=(24, 8))
                    
                    if initial_top_down_map is not None and isinstance(initial_top_down_map, np.ndarray):
                        # Create three subplots: RGB, map, and annotated RGB
                        self.ax1 = self.fig.add_subplot(131)
                        self.ax2 = self.fig.add_subplot(132)
                        self.ax3 = self.fig.add_subplot(133)
                        self.ax1.imshow(observations['rgb'])
                        self.ax1.set_title('RGB View', fontsize=10)
                        self.ax1.axis('off')
                        # Check if it's 2D or 3D array
                        if len(initial_top_down_map.shape) == 2:
                            self.ax2.imshow(initial_top_down_map, cmap='gray')
                        else:
                            self.ax2.imshow(initial_top_down_map)
                        self.ax2.set_title('Top-Down Map', fontsize=10)
                        self.ax2.axis('off')
                        # Third subplot for annotated image (initially same as RGB)
                        self.ax3.imshow(observations['rgb'])
                        self.ax3.set_title('RGB with Pixel Goal', fontsize=10)
                        self.ax3.axis('off')
                        # Add full instruction to title (no truncation)
                        self.fig.suptitle(f'Episode: {scene_id}_{episode_id:04d} - Initializing...\nInstruction: {episode_instruction}', 
                                         fontsize=9, y=0.98)
                    else:
                        # Only RGB if no map available
                        self.ax = self.fig.add_subplot(111)
                        self.ax.imshow(observations['rgb'])
                        self.ax.set_title(f'Episode: {scene_id}_{episode_id:04d} - Initializing...\nInstruction: {episode_instruction}', 
                                         fontsize=9)
                        self.ax.axis('off')
                    
                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.01)
                    print("Visualization window opened (Matplotlib)")
                except Exception as e:
                    print(f"Warning: Failed to open matplotlib window: {e}")

            vis_frames = []
            combined_frames = []  # List to store combined frames (RGB + map + annotated RGB)
            step_id = 0

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
                os.makedirs(os.path.join(self.output_path, f'combined_{self.epoch}', f'{scene_id}'), exist_ok=True)
            
            # Create directory for step images
            step_images_dir = os.path.join(
                self.output_path, 
                f'step_images_{scene_id}_{episode_id:04d}'
            )
            os.makedirs(step_images_dir, exist_ok=True)
            
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            output_ids = None

            goal = None
            action = None
            messages = []
            local_actions = []
            pixel_goal = None  # Initialize pixel_goal for visualization
            last_pixel_goal = None  # Track last pixel goal to detect updates
            last_annotated_rgb = None  # Store last annotated RGB image for OpenCV

            # List to store step-by-step data (position and LLM outputs)
            step_data_list = []

            done = False

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                
                # Collect position information for every step
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                agent_state = self.env._env.sim.get_agent_state()
                
                # Save observation image for this step
                rgb_image_filename = f'step_{step_id:04d}_rgb.jpg'
                rgb_image_path = os.path.join(step_images_dir, rgb_image_filename)
                Image.fromarray(rgb).save(rgb_image_path)
                
                # Use relative path from output_path
                rgb_image_rel_path = os.path.join(
                    f'step_images_{scene_id}_{episode_id:04d}',
                    rgb_image_filename
                )
                
                step_data = {
                    "step_id": step_id,
                    "position": {
                        "x": float(agent_state.position[0]),
                        "y": float(agent_state.position[1]),
                        "z": float(agent_state.position[2]),
                    },
                    "rotation": {
                        "w": float(agent_state.rotation.w),
                        "x": float(agent_state.rotation.x),
                        "y": float(agent_state.rotation.y),
                        "z": float(agent_state.rotation.z),
                    },
                    "gps": {
                        "x": float(x),
                        "y": float(y),
                    },
                    "camera_yaw": float(camera_yaw),
                    "rgb_image": rgb_image_rel_path,  # Relative path to saved RGB image
                    "llm_output": None,  # Will be filled if LLM generates output
                }
                
                # Get info for top-down map
                info = self.env.get_metrics()
                top_down_map = info.get('top_down_map', None)
                # Handle case where top_down_map might be a dict or None
                if top_down_map is not None and isinstance(top_down_map, dict):
                    # If it's a dict, try to get the actual map array
                    top_down_map = top_down_map.get('map', None) if isinstance(top_down_map, dict) else top_down_map
                # Ensure it's a numpy array if not None
                if top_down_map is not None and not isinstance(top_down_map, np.ndarray):
                    try:
                        top_down_map = np.array(top_down_map)
                    except Exception:
                        top_down_map = None
                
                # Display visualization window with RGB and top-down map
                if self.use_cv2:
                    try:
                        # Convert RGB to BGR for OpenCV display
                        vis_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        
                        # Add text overlay with step info
                        y_offset = 30
                        cv2.putText(vis_img, f'Step: {step_id}', (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                        if action is not None:
                            action_names = ['stop', 'move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down']
                            action_name = action_names[action] if action < len(action_names) else f'action_{action}'
                            cv2.putText(vis_img, f'Action: {action_name}', (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            y_offset += 30
                        
                        # Display full instruction (with word wrapping, no truncation)
                        instruction_text = episode_instruction  # Show full instruction
                        words = instruction_text.split()
                        lines = []
                        current_line = ""
                        for word in words:
                            test_line = current_line + " " + word if current_line else word
                            if len(test_line) > 60:  # Max characters per line
                                if current_line:
                                    lines.append(current_line)
                                current_line = word
                            else:
                                current_line = test_line
                        if current_line:
                            lines.append(current_line)
                        
                        # Display all lines (no limit, but check image bounds)
                        for i, line in enumerate(lines):
                            if y_offset + i * 25 < rgb.shape[0] - 20:  # Don't overflow image
                                cv2.putText(vis_img, line, (10, y_offset + i * 25), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Create annotated RGB image (third image) - only update when pixel_goal changes
                        pixel_goal_updated = (pixel_goal is not None and pixel_goal != last_pixel_goal)
                        
                        if pixel_goal_updated:
                            # Update annotated RGB when pixel_goal changes
                            annotated_rgb = rgb.copy()
                            try:
                                annotated_rgb_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                                y_pixel, x_pixel = pixel_goal[0], pixel_goal[1]
                                # Check if pixel coordinates are valid
                                if 0 <= y_pixel < rgb.shape[0] and 0 <= x_pixel < rgb.shape[1]:
                                    # Draw a circle at the pixel goal location
                                    cv2.circle(annotated_rgb_bgr, (x_pixel, y_pixel), 10, (0, 0, 255), 3)  # Red circle
                                    cv2.circle(annotated_rgb_bgr, (x_pixel, y_pixel), 3, (255, 255, 255), -1)  # White center
                                    # Draw coordinate text near the point
                                    coord_text = f'({x_pixel}, {y_pixel})'
                                    text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    text_x = max(0, min(x_pixel - text_size[0] // 2, rgb.shape[1] - text_size[0]))
                                    text_y = max(text_size[1] + 5, y_pixel - 15)
                                    # Draw background for text
                                    cv2.rectangle(annotated_rgb_bgr, 
                                                 (text_x - 5, text_y - text_size[1] - 5),
                                                 (text_x + text_size[0] + 5, text_y + 5),
                                                 (0, 0, 0), -1)
                                    cv2.putText(annotated_rgb_bgr, coord_text, (text_x, text_y), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Cyan text
                                    annotated_rgb = cv2.cvtColor(annotated_rgb_bgr, cv2.COLOR_BGR2RGB)
                                    last_pixel_goal = pixel_goal.copy() if isinstance(pixel_goal, list) else pixel_goal
                                    last_annotated_rgb = annotated_rgb.copy()  # Store for next iteration
                            except Exception as e:
                                if step_id == 0:
                                    print(f"Warning: Failed to draw pixel goal: {e}")
                                annotated_rgb = rgb.copy()  # Fallback to current RGB
                        else:
                            # Use last annotated RGB if available, otherwise use current RGB
                            if last_annotated_rgb is not None:
                                annotated_rgb = last_annotated_rgb.copy()
                            else:
                                annotated_rgb = rgb.copy()
                        
                        # Combine RGB, top-down map, and annotated RGB (three images)
                        images_to_concat = [vis_img]  # Start with RGB image
                        
                        if top_down_map is not None and isinstance(top_down_map, np.ndarray):
                            try:
                                # Keep map at reasonable size
                                map_height = rgb.shape[0]
                                if len(top_down_map.shape) >= 2:
                                    aspect_ratio = top_down_map.shape[1] / top_down_map.shape[0]
                                    map_width = int(map_height * aspect_ratio)
                                    max_map_width = rgb.shape[1]  # Same width as RGB
                                    if map_width > max_map_width:
                                        map_width = int(max_map_width)
                                        map_height = int(map_width / aspect_ratio)
                                    top_down_resized = cv2.resize(top_down_map, (map_width, map_height))
                                    if len(top_down_resized.shape) == 2:
                                        top_down_resized = cv2.cvtColor(top_down_resized, cv2.COLOR_GRAY2BGR)
                                    if top_down_resized.shape[0] < rgb.shape[0]:
                                        pad_height = rgb.shape[0] - top_down_resized.shape[0]
                                        top_down_resized = cv2.copyMakeBorder(top_down_resized, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                                    images_to_concat.append(top_down_resized)
                            except Exception as e:
                                if step_id == 0:
                                    print(f"Warning: Failed to process top-down map: {e}")
                        
                        # Add annotated RGB as third image
                        annotated_rgb_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                        images_to_concat.append(annotated_rgb_bgr)
                        
                        # Concatenate all images horizontally
                        vis_img = np.hstack(images_to_concat)
                        
                        cv2.imshow('Habitat Navigation', vis_img)
                        key = cv2.waitKey(1) & 0xFF  # Non-blocking wait
                        if key == ord('q'):  # Press 'q' to quit
                            print("Visualization window closed by user")
                            break
                    except Exception as e:
                        if step_id == 0:
                            print(f"OpenCV display failed: {e}")
                        pass
                elif self.use_matplotlib:
                    try:
                        action_text = ""
                        if action is not None:
                            action_names = ['stop', 'move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down']
                            action_text = action_names[action] if action < len(action_names) else f'action_{action}'
                        
                        # Create combined visualization with RGB, top-down map, and annotated RGB
                        if top_down_map is not None and isinstance(top_down_map, np.ndarray):
                            try:
                                # Check if pixel_goal was updated
                                pixel_goal_updated = (pixel_goal is not None and pixel_goal != last_pixel_goal)
                                
                                # Only clear and recreate if needed (for first time or if layout changed)
                                if not hasattr(self, 'ax1') or not hasattr(self, 'ax2') or not hasattr(self, 'ax3'):
                                    self.fig.clear()
                                    # Create three subplots: RGB, map, and annotated RGB
                                    self.ax1 = self.fig.add_subplot(131)
                                    self.ax2 = self.fig.add_subplot(132)
                                    self.ax3 = self.fig.add_subplot(133)
                                
                                # Update RGB view (always)
                                self.ax1.clear()
                                self.ax1.imshow(rgb)
                                self.ax1.set_title('RGB View', fontsize=10)
                                self.ax1.axis('off')
                                
                                # Update top-down map (always)
                                self.ax2.clear()
                                if len(top_down_map.shape) == 2:
                                    self.ax2.imshow(top_down_map, cmap='gray')
                                else:
                                    self.ax2.imshow(top_down_map)
                                self.ax2.set_title('Top-Down Map', fontsize=10)
                                self.ax2.axis('off')
                                
                                # Update annotated RGB only when pixel_goal changes
                                if pixel_goal_updated:
                                    self.ax3.clear()
                                    self.ax3.imshow(rgb)
                                    try:
                                        y_pixel, x_pixel = pixel_goal[0], pixel_goal[1]
                                        if 0 <= y_pixel < rgb.shape[0] and 0 <= x_pixel < rgb.shape[1]:
                                            self.ax3.plot(x_pixel, y_pixel, 'ro', markersize=12, markeredgewidth=2, markeredgecolor='red', markerfacecolor='white')
                                            coord_text = f'({x_pixel}, {y_pixel})'
                                            self.ax3.text(x_pixel, y_pixel - 20, coord_text, 
                                                       fontsize=10, color='cyan', weight='bold',
                                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                                        last_pixel_goal = pixel_goal.copy() if isinstance(pixel_goal, list) else pixel_goal
                                    except Exception as e:
                                        if step_id == 0:
                                            print(f"Warning: Failed to draw pixel goal in matplotlib: {e}")
                                    self.ax3.set_title('RGB with Pixel Goal', fontsize=10)
                                    self.ax3.axis('off')
                                    # Force redraw of third subplot
                                    plt.draw()
                            except Exception as e:
                                # If map display fails, just show RGB
                                if step_id == 0:
                                    print(f"Warning: Failed to display top-down map: {e}")
                                self.fig.clear()
                                self.ax = self.fig.add_subplot(111)
                                self.ax.imshow(rgb)
                                # Draw pixel goal on image if available
                                if pixel_goal is not None:
                                    try:
                                        y_pixel, x_pixel = pixel_goal[0], pixel_goal[1]
                                        if 0 <= y_pixel < rgb.shape[0] and 0 <= x_pixel < rgb.shape[1]:
                                            self.ax.plot(x_pixel, y_pixel, 'ro', markersize=12, markeredgewidth=2, markeredgecolor='red', markerfacecolor='white')
                                            coord_text = f'({x_pixel}, {y_pixel})'
                                            self.ax.text(x_pixel, y_pixel - 20, coord_text, 
                                                       fontsize=10, color='cyan', weight='bold',
                                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                                    except Exception:
                                        pass
                                self.ax.set_title(f'Step: {step_id} | Action: {action_text} | Episode: {scene_id}_{episode_id:04d}\nInstruction: {episode_instruction}', 
                                                 fontsize=9)
                                self.ax.axis('off')
                            
                            # Add full instruction to title (no truncation)
                            self.fig.suptitle(f'Step: {step_id} | Action: {action_text} | Episode: {scene_id}_{episode_id:04d}\nInstruction: {episode_instruction}', 
                                             fontsize=9, y=0.98)
                        else:
                            # Only RGB if no map available
                            self.ax = self.fig.add_subplot(111)
                            self.ax.imshow(rgb)
                            # Draw pixel goal on image if available
                            if pixel_goal is not None:
                                try:
                                    y_pixel, x_pixel = pixel_goal[0], pixel_goal[1]
                                    if 0 <= y_pixel < rgb.shape[0] and 0 <= x_pixel < rgb.shape[1]:
                                        self.ax.plot(x_pixel, y_pixel, 'ro', markersize=12, markeredgewidth=2, markeredgecolor='red', markerfacecolor='white')
                                        coord_text = f'({x_pixel}, {y_pixel})'
                                        self.ax.text(x_pixel, y_pixel - 20, coord_text, 
                                                   fontsize=10, color='cyan', weight='bold',
                                                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                                except Exception:
                                    pass
                            self.ax.set_title(f'Step: {step_id} | Action: {action_text} | Episode: {scene_id}_{episode_id:04d}\nInstruction: {episode_instruction}', 
                                             fontsize=9)
                            self.ax.axis('off')
                        
                        plt.tight_layout()
                        plt.draw()
                        plt.pause(0.01)  # Small pause to update display
                        
                        # Save matplotlib figure as image for combined video
                        if self.save_video:
                            try:
                                # Get figure as numpy array
                                self.fig.canvas.draw()
                                fig_data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                                fig_data = fig_data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                                combined_frames.append(fig_data)
                            except Exception as e:
                                if step_id == 0:
                                    print(f"Warning: Failed to save matplotlib figure: {e}")
                    except Exception as e:
                        if step_id == 0:
                            print(f"Matplotlib display failed: {e}")
                        pass
                # x, y and camera_yaw already collected above
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    self.xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
                    @ self.get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                save_dot = False
                if action == 5:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = self.preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    print(f"[habitat_vln_evaluator] Image size before resize: {image.size} (W x H)")
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                    if self.model_args.mode == 'dual_system':
                        down_observations, _, done, _ = self.env.step(5)
                        down_observations, _, done, _ = self.env.step(5)

                        look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                        depth = down_observations["depth"]
                        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                        depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                        depth = depth * 1000
                        look_down_depth, resize_shape = self.preprocess_depth_image_v2(
                            Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                            do_depth_scale=True,
                            depth_scale=1000,
                            target_height=224,
                            target_width=224,
                        )
                        look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                        look_down_depth[look_down_depth > 5.0] = 5.0

                        self.env.step(4)
                        self.env.step(4)

                info = self.env.get_metrics()

                if len(action_seq) == 0 and goal is None:
                    if action != 5:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        # Add fixed demo reference text prompt (no image)
                        sources[0]["value"] += f' Navigation tip (from previous demonstration): Do not enter the kitchen.'
                        
                        # Build input images from history and current observations
                        history_id = sorted(history_id)
                        print('history_idddddddd', step_id, history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        print(f"[evaluator] step_id {step_id} Total images: {len(input_images)}")
                        input_img_id = 0
                    else:
                        assert action == 5
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        # messages.append(
                        #     {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        # )
                        input_img_id = -1

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    print('sources', step_id, sources)
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    print('step_id', step_id, 'messages:', messages)

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)
                    
                    # Update step data with LLM output
                    step_data["llm_output"] = llm_outputs

                    if bool(re.search(r'\d', llm_outputs)):
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                        pixel_goal = [int(coord[1]), int(coord[0])]

                        intrinsic_matrix = self.get_intrinsic_matrix(
                            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
                        )
                        goal = self.pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)
                        print('before', goal, depth.shape)
                        goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                        if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                            goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                        # look down --> horizontal
                        self.env.step(4)
                        self.env.step(4)

                        # Forking logic based on mode
                        if self.model_args.mode == 'system2':
                            action = agent.get_next_action(goal)
                            if action == 0:
                                goal = None
                                output_ids = None
                                action = 2  # random action
                                print('conduct a random action 2')
                                observations, _, done, _ = self.env.step(action)
                                step_data_list.append(step_data)
                                step_id += 1
                                messages = []
                                continue
                        else:  # dual-system logic
                            local_actions = []
                            pixel_values = inputs.pixel_values
                            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                            with torch.no_grad():
                                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                            # prepocess align with navdp
                            image_dp = (
                                torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                            )
                            pix_goal_image = copy.copy(image_dp)
                            images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                            depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                            pix_goal_depth = copy.copy(depth_dp)
                            depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                            with torch.no_grad():
                                dp_actions = self.model.generate_traj(
                                    traj_latents, images_dp, depths_dp, use_async=True
                                )

                            random_choice = np.random.choice(dp_actions.shape[0])
                            if self.model_args.continuous_traj:
                                action_list = traj_to_actions(dp_actions)
                                if len(action_list) < 8:
                                    action_list += [0] * (8 - len(action_list))
                            else:
                                action_list = chunk_token(dp_actions[random_choice])

                            local_actions = action_list
                            if len(local_actions) >= 4:
                                local_actions = local_actions[:4]
                            action = local_actions[0]
                            if action == 0:
                                goal = None
                                output_ids = None
                                action = 2  # random action
                                print('conduct a random action 2')
                                observations, _, done, _ = self.env.step(action)
                                step_data_list.append(step_data)
                                step_id += 1
                                messages = []
                                continue

                        print('predicted goal', pixel_goal, goal, flush=True)
                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    # Forking logic based on mode
                    if self.model_args.mode == 'system2':
                        action = agent.get_next_action(goal)
                        action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                        action = action[0] if hasattr(action, "__len__") else action
                    else:  # dual-system logic
                        if len(local_actions) == 0:
                            # navdp
                            local_actions = []
                            image_dp = (
                                torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                            )

                            images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                            depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                            depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                dp_actions = self.model.generate_traj(
                                    traj_latents, images_dp, depths_dp, use_async=True
                                )

                            random_choice = np.random.choice(dp_actions.shape[0])
                            if self.model_args.continuous_traj:
                                action_list = traj_to_actions(dp_actions)
                                if len(action_list) < 8:
                                    action_list += [0] * (8 - len(action_list))
                            else:
                                action_list = chunk_token(dp_actions[random_choice])
                            print("first action_list", action_list)

                            local_actions = action_list
                            if len(local_actions) >= 4:
                                local_actions = local_actions[:4]
                            # if len(local_actions) >= 2:
                            #     local_actions = local_actions[:2]

                            print("local_actions", local_actions)

                            action = local_actions.pop(0)
                            # navdp
                        else:
                            action = local_actions.pop(0)

                    forward_action += 1
                    print('forward_action', forward_action, flush=True)
                    if forward_action > 8:
                        goal = None
                        output_ids = None
                        messages = []
                        step_data_list.append(step_data)
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == 0:
                        goal = None
                        output_ids = None
                        messages = []
                        step_data_list.append(step_data)
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0
                    # Reset pixel_goal when action is 0 (stop)
                    pixel_goal = None
                
                # Append step data to list (with or without LLM output)
                step_data_list.append(step_data)

                if info['top_down_map'] is not None:
                    if save_dot:
                        save_raw_image = self.dot_matrix_two_dimensional(
                            save_raw_image, save_img=False, save_path=f'test_{step_id}.jpg', pixel_goal=pixel_goal
                        )
                    if self.save_video:
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                        vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                # Render visualization window
                try:
                    self.env.render()
                except Exception as e:
                    # If rendering fails, continue without visualization
                    pass

                # refactor: core
                if action == 5:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []

            # ---------- 3. End of episode -----------
            # Update result and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode result.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            
            # Save step-by-step data (position and LLM outputs) to JSON
            step_data_file = os.path.join(
                self.output_path, 
                f'step_data_{scene_id}_{episode_id:04d}.json'
            )
            episode_step_data = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "episode_instruction": episode_instruction,
                "total_steps": step_id,
                "steps": step_data_list,
            }
            with open(step_data_file, 'w', encoding='utf-8') as f:
                json.dump(episode_step_data, f, indent=2, ensure_ascii=False)
            print(f"Step-by-step data saved to: {step_data_file}")
            if self.save_video:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
                # Save combined video (RGB + top-down map + annotated RGB)
                if combined_frames:
                    images_to_video(
                        combined_frames,
                        os.path.join(self.output_path, f'combined_{self.epoch}', f'{scene_id}'),
                        f'{episode_id:04d}',
                        fps=6,
                        quality=9,
                    )
            vis_frames.clear()
            combined_frames.clear()

        self.env.close()
        
        # Close visualization windows
        if hasattr(self, 'use_cv2') and self.use_cv2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if hasattr(self, 'use_matplotlib') and self.use_matplotlib:
            try:
                plt.close('all')
            except Exception:
                pass

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def preprocess_depth_image_v2(
        self, depth_image, do_depth_scale=True, depth_scale=1000, target_height=None, target_width=None
    ):
        if target_height is None:
            target_height = self.image_processor.crop_size['height']  # 384
            target_width = self.image_processor.crop_size['width']  # 384

        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)

        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale

        return img, (target_width, target_height)

    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        return intrinsic_matrix

    def get_axis_align_matrix(self):
        ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        return ma

    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def xyz_pitch_to_tf_matrix(self, xyz: np.ndarray, pitch: float) -> np.ndarray:
        """Converts a given position and pitch angle to a 4x4 transformation matrix.

        Args:
            xyz (np.ndarray): A 3D vector representing the position.
            pitch (float): The pitch angle in radians for y axis.
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """

        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch), x],
                [0, 1, 0, y],
                [-np.sin(pitch), 0, np.cos(pitch), z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def xyz_yaw_pitch_to_tf_matrix(self, xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        """Converts a given position and yaw, pitch angles to a 4x4 transformation matrix.

        Args:
            xyz (np.ndarray): A 3D vector representing the position.
            yaw (float): The yaw angle in radians.
            pitch (float): The pitch angle in radians for y axis.
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        x, y, z = xyz
        rot1 = self.xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
        rot2 = self.xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rot1 @ rot2
        transformation_matrix[:3, 3] = xyz
        return transformation_matrix

    def pixel_to_gps(self, pixel, depth, intrinsic, tf_camera_to_episodic):
        '''
        Args:
            pixel: (2,) - [u, v] pixel coordinates
            depth: (H, W) - depth image where depth[v, u] gives depth in meters
            intrinsic: (4, 4) - camera intrinsic matrix
            tf_camera_to_episodic: (4, 4) - transformation from camera to episodic frame
        Returns:
            (x, y): (x, y) coordinates in the episodic frame
        '''
        v, u = pixel
        z = depth[v, u]
        print("depthhhhhhhhhhhhhh", z)

        x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
        point_camera = np.array([x, y, z, 1.0])

        # Transform to episodic frame
        point_episodic = tf_camera_to_episodic @ point_camera
        point_episodic = point_episodic[:3] / point_episodic[3]

        x = point_episodic[0]
        y = point_episodic[1]

        return (x, y)  # same as habitat gps

    def dot_matrix_two_dimensional(
        self,
        image_or_image_path,
        save_path=None,
        dots_size_w=8,
        dots_size_h=8,
        save_img=False,
        font_path='fonts/arial.ttf',
        pixel_goal=None,
    ):
        """
        takes an original image as input, save the processed image to save_path. Each dot is labeled with two-dimensional Cartesian coordinates (x,y). Suitable for single-image tasks.
        control args:
        1. dots_size_w: the number of columns of the dots matrix
        2. dots_size_h: the number of rows of the dots matrix
        """
        with open_image(image_or_image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            draw = ImageDraw.Draw(img, 'RGB')

            width, height = img.size
            grid_size_w = dots_size_w + 1
            grid_size_h = dots_size_h + 1
            cell_width = width / grid_size_w
            cell_height = height / grid_size_h

            font = ImageFont.truetype(font_path, width // 40)  # Adjust font size if needed; default == width // 40

            target_i = target_j = None
            if pixel_goal is not None:
                y_pixel, x_pixel = pixel_goal[0], pixel_goal[1]
                # Validate pixel coordinates
                if not (0 <= x_pixel < width and 0 <= y_pixel < height):
                    raise ValueError(f"pixel_goal {pixel_goal} exceeds image dimensions ({width}x{height})")

                # Convert to grid coordinates
                target_i = round(x_pixel / cell_width)
                target_j = round(y_pixel / cell_height)

                # Validate grid bounds
                if not (1 <= target_i <= dots_size_w and 1 <= target_j <= dots_size_h):
                    raise ValueError(
                        f"pixel_goal {pixel_goal} maps to grid ({target_j},{target_i}), "
                        f"valid range is (1,1)-({dots_size_h},{dots_size_w})"
                    )

            count = 0

            for j in range(1, grid_size_h):
                for i in range(1, grid_size_w):
                    x = int(i * cell_width)
                    y = int(j * cell_height)

                    pixel_color = img.getpixel((x, y))
                    # choose a more contrasting color from black and white
                    if pixel_color[0] + pixel_color[1] + pixel_color[2] >= 255 * 3 / 2:
                        opposite_color = (0, 0, 0)
                    else:
                        opposite_color = (255, 255, 255)

                    if pixel_goal is not None and i == target_i and j == target_j:
                        opposite_color = (255, 0, 0)  # Red for target

                    circle_radius = width // 240  # Adjust dot size if needed; default == width // 240
                    draw.ellipse(
                        [(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
                        fill=opposite_color,
                    )

                    text_x, text_y = x + 3, y
                    count_w = count // dots_size_w
                    count_h = count % dots_size_w
                    label_str = f"({count_w+1},{count_h+1})"
                    draw.text((text_x, text_y), label_str, fill=opposite_color, font=font)
                    count += 1
            if save_img:
                print(">>> dots overlaid image processed, stored in", save_path)
                img.save(save_path)
            return img
