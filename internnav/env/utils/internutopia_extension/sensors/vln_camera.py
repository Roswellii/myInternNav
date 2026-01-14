from typing import Dict

import numpy as np
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.sensor.camera import ICamera
from internutopia.core.sensor.sensor import BaseSensor

from internnav.utils.common_log_util import common_logger as log

from ..configs.sensors.vln_camera import VLNCameraCfg


@BaseSensor.register('VLNCamera')
class VLNCamera(BaseSensor):
    """
    wrap of isaac sim's Camera class
    """

    def __init__(self, config: VLNCameraCfg, robot: BaseRobot, scene: IScene):
        super().__init__(config, robot, scene)
        self.config = config
        self._camera = None

    def get_data(self) -> Dict:
        output_data = {}
        try:
            output_data['rgba'] = self._camera.get_rgba()
            rgba_shape = output_data['rgba'].shape
            log.info(f"[VLNCamera] RGBA shape: {rgba_shape}, expected: ({self.config.resolution[1]}, {self.config.resolution[0]}, 4)")
            
            if output_data['rgba'].shape[0] != self.config.resolution[1]:
                log.error(f"[VLNCamera] RGBA shape mismatch! Got {rgba_shape}, expected height {self.config.resolution[1]}")
                log.error(f"[VLNCamera] Using random data as fallback!")
                output_data['rgba'] = np.random.randint(
                    0, 256, (self.config.resolution[1], self.config.resolution[0], 4), dtype=np.uint8
                )
            else:
                rgba_stats = {
                    'min': float(output_data['rgba'].min()),
                    'max': float(output_data['rgba'].max()),
                    'mean': float(output_data['rgba'].mean()),
                }
                log.info(f"[VLNCamera] RGBA stats: {rgba_stats}")
                if rgba_stats['max'] == 0.0:
                    log.warning(f"[VLNCamera] WARNING: RGBA image is completely black (all zeros)!")
            
            output_data['depth'] = self._camera.get_distance_to_image_plane()
            depth_shape = output_data['depth'].shape
            log.info(f"[VLNCamera] Depth shape: {depth_shape}, expected: ({self.config.resolution[1]}, {self.config.resolution[0]})")
            
            if output_data['depth'].shape[0] != self.config.resolution[1]:
                log.error(f"[VLNCamera] Depth shape mismatch! Got {depth_shape}, expected height {self.config.resolution[1]}")
                log.error(f"[VLNCamera] Using random data as fallback!")
                output_data['depth'] = np.random.uniform(
                    0, 256, size=(self.config.resolution[1], self.config.resolution[0])
                ).astype(np.float32)
            else:
                depth_stats = {
                    'min': float(output_data['depth'].min()),
                    'max': float(output_data['depth'].max()),
                    'mean': float(output_data['depth'].mean()),
                }
                log.info(f"[VLNCamera] Depth stats: {depth_stats}")
        except Exception as e:
            log.error(f"[VLNCamera] Error getting camera data: {e}")
            import traceback
            log.error(traceback.format_exc())
            raise
        
        return self._make_ordered(output_data)

    def post_reset(self):
        self.restore_sensor_info()

    def restore_sensor_info(self):
        self.cleanup()
        prim_path = self._robot.config.prim_path + '/' + self.config.prim_path
        log.info(f"[VLNCamera] Creating camera: {self.config.name}")
        log.info(f"[VLNCamera] Prim path: {prim_path}")
        log.info(f"[VLNCamera] Resolution: {self.config.resolution}")
        
        _camera = ICamera.create(
            name=self.config.name,
            prim_path=prim_path,
            rgba=True,
            bounding_box_2d_tight=False,
            distance_to_image_plane=True,
            camera_params=False,
            resolution=self.config.resolution,
        )
        self._camera: ICamera = _camera
        log.info(f"[VLNCamera] Camera created successfully")

    def cleanup(self) -> None:
        if self._camera is not None:
            self._camera.cleanup()

    def set_world_pose(self, *args, **kwargs):
        self._camera.set_world_pose(*args, **kwargs)

    def get_world_pose(self):
        return self._camera.get_world_pose()
