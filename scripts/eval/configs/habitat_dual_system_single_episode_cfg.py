from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",  # inference mode: dual_system or system2
            "model_path": "checkpoints/InternVLA-N1-wo-dagger",  # path to model checkpoint
            "num_future_steps": 4,  # number of future steps for prediction
            "num_frames": 32,  # number of frames used in evaluation
            "num_history": 8,
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            "predict_step_nums": 32,  # number of steps to predict
            "continuous_traj": True,  # whether to use continuous trajectory
            "max_new_tokens": 1024,  # maximum number of tokens for generation
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
            # Filter to specific episodes only
            # Format: list of dicts with 'scene_id' and 'episode_id', or list of tuples (scene_id, episode_id)
            'filter_episodes': [
                {'scene_id': '2azQ1b91cZZ', 'episode_id': 71}
            ],
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        # all current parse args
        "output_path": "./logs/habitat/test_dual_system_0126_demo_text8",  # output directory for logs/results
        "save_video": True,  # whether to save videos
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        "camera_height": 0.2,  # Camera height in meters (y-axis), set to 0.2m (low perspective)
        # distributed settings
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
    },
)

