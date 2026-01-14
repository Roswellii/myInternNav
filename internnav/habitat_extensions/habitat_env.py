import json
import os
from typing import Any, Dict, List, Optional

# 在导入habitat之前设置EGL环境变量以使用NVIDIA GPU渲染
# 覆盖conda环境中的Mesa配置，使用系统NVIDIA驱动
os.environ['__EGL_VENDOR_LIBRARY_DIRS'] = '/usr/share/glvnd/egl_vendor.d'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
# 设置EGL设备
if 'EGL_VISIBLE_DEVICES' not in os.environ:
    os.environ['EGL_VISIBLE_DEVICES'] = '0'
# 确保CUDA设备可见
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 设置库路径优先使用系统的NVIDIA库
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ['LD_LIBRARY_PATH']
else:
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import base


@base.Env.register('habitat')
class HabitatEnv(base.Env):
    def __init__(self, env_config: EnvCfg, task_config: TaskCfg):
        """
        env_settings include:
            - habitat_config: loaded from get_habitat_config
            - rank: int, rank index for sharding
            - world_size: int, total number of ranks
        """
        try:
            from habitat import Env
        except ImportError as e:
            raise RuntimeError(
                "Habitat modules could not be imported. " "Make sure both repositories are installed and on PYTHONPATH."
            ) from e

        super().__init__(env_config, task_config)

        self.config = env_config.env_settings['habitat_config']
        
        # 尝试创建环境，如果失败则尝试修改配置
        try:
            self._env = Env(self.config)
        except Exception as e:
            error_msg = str(e)
            # 如果是OpenGL相关错误，尝试禁用渲染器
            if "OpenGL" in error_msg or "GL::Context" in error_msg or "EGL" in error_msg:
                print(f"警告: OpenGL初始化失败 ({error_msg[:100]})，尝试禁用渲染器...")
                try:
                    with habitat.config.read_write(self.config):
                        if hasattr(self.config.habitat.simulator, 'habitat_sim_v0'):
                            if hasattr(self.config.habitat.simulator.habitat_sim_v0, 'create_renderer'):
                                self.config.habitat.simulator.habitat_sim_v0.create_renderer = False
                            if hasattr(self.config.habitat.simulator.habitat_sim_v0, 'gpu_device_id'):
                                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
                    self._env = Env(self.config)
                    print("✓ 使用无渲染器模式成功创建环境（无可视化）")
                except Exception as e2:
                    raise RuntimeError(
                        f"无法创建Habitat环境。OpenGL错误: {error_msg[:200]}。"
                        f"尝试禁用渲染器也失败: {str(e2)[:200]}。"
                        "建议：1) 重新安装habitat-sim 2) 使用Docker环境 3) 检查GPU驱动"
                    ) from e2
            else:
                raise

        self.rank = env_config.env_settings.get('rank', 0)
        self.world_size = env_config.env_settings.get('world_size', 1)
        self._current_episode_index: int = 0
        self._last_obs: Optional[Dict[str, Any]] = None

        self.is_running = True
        self.output_path = env_config.env_settings.get('output_path', './output')

        # generate episodes
        # self._env.episodes = self._env.episodes[0:1]  # for debug
        self.episodes = self.generate_episodes()
        # print(self.episodes)

    def generate_episodes(self) -> List[Any]:
        """
        Generate list of episodes for the current split, already:
        - grouped by scene
        - filtered by done_res (the path is self.output_path/progress.json)
        - sharded by (rank, world_size)
        """
        all_episodes = []

        # group episodes by scene
        scene_episode_dict: Dict[str, List[Any]] = {}
        for episode in self._env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)

        # load done_res
        done_res = set()
        result_path = os.path.join(self.output_path, 'progress.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                for line in f:
                    res = json.loads(line)
                    # only skip if current format has scene_id
                    if "scene_id" in res:
                        done_res.add((res["scene_id"], res["episode_id"]))

        # iterate scenes in order, collect all episodes
        for scene in sorted(scene_episode_dict.keys()):
            per_scene_eps = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]

            # shard by rank index / world_size
            for episode in per_scene_eps[self.rank :: self.world_size]:
                episode_id = int(episode.episode_id)
                if (scene_id, episode_id) in done_res:
                    continue
                all_episodes.append(episode)

        return all_episodes

    def reset(self):
        """
        load next episode and return first observation
        """
        # no more episodes
        if not (0 <= self._current_episode_index < len(self.episodes)):
            self.is_running = False
            return

        # Manually set to next episode in habitat
        self._env.current_episode = self.episodes[self._current_episode_index]
        self._current_episode_index += 1

        # Habitat reset
        self._last_obs = self._env.reset()

        return self._last_obs

    def step(self, action: List[Any]):
        """
        step the environment with given action

        Args: action: List[Any], action for each env in the batch

        Return: obs, reward, done, info
        """
        obs = self._env.step(action)
        done = self._env.episode_over
        info = self._env.get_metrics()
        reward = info.get('reward', 0.0)
        return obs, reward, done, info

    def close(self):
        print('Habitat Env close')
        self._env.close()

    def render(self):
        self._env.render()

    def get_observation(self) -> Dict[str, Any]:
        return self._env.get_observations()

    def get_metrics(self) -> Dict[str, Any]:
        return self._env.get_metrics()

    def get_current_episode(self):
        return self._env.current_episode
