import copy
from typing import List
from numpy import ndarray

from rlbench.backend.const import *
from rlbench.backend.utils import task_file_to_task_class

from utils import ROLLOUT_IMAGE_SIZE, CustomRLRenchEnv2
from rlbench import CameraConfig, ObservationConfig
from pyrep.const import RenderMode
from PIL import Image
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.agents.agent import ActResult
from yarr.utils.transition import Transition
import numpy as np

from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    Scene,
)


class EndEffectorPoseViaPlanning2(EndEffectorPoseViaPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        action[:3] = np.clip(
            action[:3],
            np.array(
                [scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]
            )
            + 1e-7,
            np.array(
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]
            )
            - 1e-7,
        )

        super().action(scene, action, ignore_collisions)


CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]


STAND_POSE_ACTION = np.array([0.29791760444641113, 
                              0.08399009704589844, 
                              1.3635880947113037, 
                              -0.6755902076156602, 
                              -0.7372773368101241, 
                              4.514521653042017e-17,
                              4.1367969264590704e-17, 
                              1.0, 
                              0.0])

def create_obs_config(camera_names: List[str],
                       camera_resolution: List[int],
                       method_name: str,
                       USE_DEPTH = False,
                        USE_PC = True):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True if USE_PC else False,
        mask=False,
        depth=True if USE_DEPTH else False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL,
        depth_in_meters=True if USE_DEPTH else False,)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config


class RLBenchSim:
    def __init__(
        self,  
        task_name: str,
        dataset_root: str,
        episode_length: int=30,
        record_every_n: int = 5, # -1 means no recording
        resolution: int=ROLLOUT_IMAGE_SIZE,
        record_queue=None,
        never_terminal=False,
        unseen_task=False,
        use_depth=False,
        use_pc=False,
        use_joint_pos=None,
    ):
        self.task_name = task_name
        self.dataset_root = dataset_root
        self.episode_length = episode_length
        self.record_every_n = record_every_n
        self.record_queue = record_queue
        self.never_terminal = never_terminal
        self.unseen_task = unseen_task
        self.use_depth = use_depth
        self.use_pc = use_pc
        self.use_joint_pos = use_joint_pos
        
        self.setup_env(resolution)

        self.last_action = None

        
    def reset(self, episode_num: int = 0, not_load_image: bool = True) -> dict:
        obs_dict, obs = self.env.reset_to_demo(episode_num, not_load_image)
        self.d = self.env.d
        return obs_dict, obs
    
    def setup_env(self, resolution):
        camera_resolution = [resolution, resolution]
        obs_config = create_obs_config(CAMERAS, camera_resolution, method_name="", USE_DEPTH=self.use_depth, USE_PC=self.use_pc)

        gripper_mode = Discrete()
        if self.use_joint_pos is None:
            arm_action_mode = EndEffectorPoseViaPlanning()
        else:
            from rlbench.action_modes.arm_action_modes import JointPosition
            arm_action_mode = JointPosition()
        action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)
        self.env = CustomRLRenchEnv2(
            record_queue=self.record_queue,
            task_class=task_file_to_task_class(self.task_name),
            observation_config=obs_config,
            action_mode=action_mode,
            dataset_root=self.dataset_root,
            episode_length=self.episode_length,
            headless=True,
            time_in_state=True,
            include_lang_goal_in_obs=True,
            record_every_n=self.record_every_n,
            never_terminal=self.never_terminal,
            unseen_task=self.unseen_task,
        )
        self.env.eval = True
        self.env.launch()
    
    def set_new_task(self, task_name: str):
        self.env.set_new_task(task_name)
        self.task_name = task_name
    
    def set_new_dataset(self, dataset_root: str):
        self.env._rlbench_env._dataset_root = dataset_root
    
    @property
    def task_goal(self):
        return self.env._lang_goal

    def step(self, action: ndarray) -> Transition:
        # action is (9, ) array, 3 for pose, 4 for quaternion, 1 for gripper, 1 for ignore_collision
        wrap_action = ActResult(action=action)            
        transition = self.env.step(wrap_action) # get Transition(obs, reward, terminal, info, summaries)    
        if transition.info['error_status'] == "error":
            print(f"Error: action was {action}")
            if self.task_name in ["put_item_in_drawer"]:
                transition = self.env.step(ActResult(action=STAND_POSE_ACTION))
            if self.task_name in ["open_drawer"] and self.last_action is not None: # avoid strange invalid error
                action[0] = (self.last_action[0] + action[0])/2
                action[2] = (self.last_action[2] + action[2])/2
                transition = self.env.step(ActResult(action=action))
        if isinstance(transition, tuple):
            transition = transition[0]
        self.transition = transition
        self.last_action = action
        return transition
    
    def is_success(self) -> bool:
        # always called when simulation ends
        score = self.transition.reward
        return True if score == 100.0 else False
        
    def close(self):
        self.env.shutdown()
    
    
    def get_video_frames(self, res=128, return_pil=True):
        ret = []
        for fra in self.env._recorded_images:
            if fra.shape[0] == 3:
                fra = fra.transpose(1, 2, 0)
            fra = Image.fromarray(fra).resize((res, res))
            if return_pil:
                ret.append(fra)
            else:
                ret.append(np.array(fra))
        self.env._recorded_images.clear()
        return ret
    