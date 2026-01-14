import os
import sys
from typing import Any, Dict, List

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import base
from internnav.env.utils.episode_loader import (
    ResumablePathKeyEpisodeloader,
    generate_vln_episode,
)
from internnav.utils.common_log_util import common_logger as log


@base.Env.register('internutopia')
class InternutopiaEnv(base.Env):
    def __init__(self, env_config: EnvCfg, task_config: TaskCfg):
        try:
            from internutopia.core.config import Config, SimConfig
            from internutopia.core.config.distribution import RayDistributionCfg
            from internutopia.core.vec_env import Env

            from internnav.env.utils.internutopia_extension import import_extensions
        except ImportError as e:
            raise RuntimeError(
                "InternUtopia modules could not be imported. "
                "Make sure both repositories are installed and on PYTHONPATH."
            ) from e

        super().__init__(env_config, task_config)
        env_settings = self.env_config.env_settings
        task_settings = self.task_config.task_settings

        # Ensure logger is initialized before logging
        from internnav.utils.common_log_util import init as log_init
        if not hasattr(log_init, '_initialized'):
            log_init('internutopia_env')
            log_init._initialized = True

        log.info("=" * 80)
        log.info("[InternutopiaEnv] Starting environment initialization...")
        log.info(f"[InternutopiaEnv] env_type: {env_config.env_type}")
        log.info(f"[InternutopiaEnv] env_settings: {env_settings}")
        log.info(f"[InternutopiaEnv] task_settings: {task_settings}")

        # generate episodes
        log.info("[InternutopiaEnv] Generating episodes...")
        self.episode_loader = ResumablePathKeyEpisodeloader(
            env_settings['dataset'].dataset_type,
            **env_settings['dataset'].dataset_settings,
            rank=env_settings['rank'],
            world_size=env_settings['world_size']
        )
        self.episodes = generate_vln_episode(self.episode_loader, task_config)
        if len(self.episodes) == 0:
            log.error("No episodes found for the given configuration.")
            sys.exit(0)
        log.info(f"[InternutopiaEnv] Generated {len(self.episodes)} episodes")
        
        # Log scene information from first episode
        if len(self.episodes) > 0:
            first_episode = self.episodes[0]
            scene_path = getattr(first_episode, 'scene_asset_path', None)
            if scene_path:
                log.info(f"[InternutopiaEnv] Scene path: {scene_path}")
                if os.path.exists(scene_path):
                    log.info(f"[InternutopiaEnv] Scene file exists: {os.path.getsize(scene_path)} bytes")
                else:
                    log.error(f"[InternutopiaEnv] Scene file NOT found: {scene_path}")
        
        task_settings.update({'episodes': self.episodes})

        # set visible device for isaac sim
        cuda_device = str(env_settings.get('local_rank', 0))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        log.info(f"[InternutopiaEnv] Set CUDA_VISIBLE_DEVICES={cuda_device}")

        # Log simulator config
        log.info("[InternutopiaEnv] Creating SimConfig...")
        log.info(f"[InternutopiaEnv] SimConfig params: headless={env_settings.get('headless', None)}, "
                f"use_fabric={env_settings.get('use_fabric', None)}")
        
        # Fix viewport resolution to avoid DLSS warning (minimum 300x300, but use 640x480 to match camera)
        # The 320x240 issue might be from viewport scaling, ensure minimum resolution
        sim_config_dict = dict(env_settings)
        # Remove width/height from sim_config if they exist, as they might conflict with Isaac Sim's internal settings
        # Instead, we'll set them via environment or Isaac Sim's configuration
        if 'width' in sim_config_dict:
            del sim_config_dict['width']
        if 'height' in sim_config_dict:
            del sim_config_dict['height']
        log.info(f"[InternutopiaEnv] SimConfig params (after cleanup): {list(sim_config_dict.keys())}")
        
        config = Config(
            simulator=SimConfig(**sim_config_dict),
            env_num=task_settings['env_num'],
            env_offset_size=task_settings['offset_size'],
            task_configs=task_settings['episodes'],
        )
        if 'distribution_config' in env_settings:
            distribution_config = RayDistributionCfg(**env_settings['distribution_config'])
            config = config.distribute(distribution_config)
            log.info(f"[InternutopiaEnv] Using distributed config")

        # register all extensions
        log.info("[InternutopiaEnv] Registering extensions...")
        import_extensions()

        # Create environment (this will start Isaac Sim)
        log.info("[InternutopiaEnv] Creating InternUtopia Env (this will start Isaac Sim)...")
        log.info("[InternutopiaEnv] This may take a while...")
        self.env = Env(config)
        log.info("[InternutopiaEnv] InternUtopia Env created successfully!")
        
        # Try to get initial observation to verify rendering
        try:
            log.info("[InternutopiaEnv] Attempting to get initial observation...")
            obs = self.get_observation()
            if obs:
                log.info(f"[InternutopiaEnv] Initial observation keys: {list(obs.keys())}")
                for key, value in obs.items():
                    if hasattr(value, 'shape'):
                        log.info(f"[InternutopiaEnv] Observation '{key}' shape: {value.shape}, dtype: {value.dtype}")
                    elif isinstance(value, (list, tuple)):
                        log.info(f"[InternutopiaEnv] Observation '{key}' type: {type(value)}, length: {len(value)}")
        except Exception as e:
            log.warning(f"[InternutopiaEnv] Failed to get initial observation: {e}")
        
        log.info("=" * 80)

    def reset(self, reset_index=None):
        return self.env.reset(reset_index)

    def step(self, action: List[Any]):
        return self.env.step(action)

    def is_running(self):
        return True

    def close(self):
        print('Vln Env close')
        self.env.close()

    def render(self):
        self.env.render()

    def get_observation(self) -> Dict[str, Any]:
        return self.env.get_observations()

    def get_info(self) -> Dict[str, Any]:
        pass
