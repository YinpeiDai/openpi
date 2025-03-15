import copy
import math
import pickle
import numpy as np
from pyrep.objects import VisionSensor, Dummy
from pyrep.const import RenderMode
from rlbench.backend.conditions import Condition

from yarr.agents.agent import ActResult, VideoSummary, TextSummary
from yarr.utils.transition import Transition

from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
from yarr.utils.process_str import change_case
from rlbench.backend.utils import task_file_to_task_class
from PIL import Image
from dataclasses import dataclass, field
from numpy import ndarray
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
from rlbench.backend.observation import Observation
from scipy.spatial.transform import Rotation as R

from typing import Type, List

import numpy as np
from rlbench import ObservationConfig, ActionMode
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from yarr.agents.agent import ActResult, VideoSummary, TextSummary
from yarr.envs.rlbench_env import RLBenchEnv, MultiTaskRLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from yarr.utils.process_str import change_case

from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy


RLBENCH_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]



def quat2rpy(action):
    pos = action[:3]
    quat = action[3:7]
    rpy = R.from_quat(quat).as_euler("xyz")
    return np.concatenate([pos, rpy])

def rpy2quat(action):
    pos = action[:3]
    rpy = action[3:6]
    quat = R.from_euler("xyz", rpy).as_quat()
    return np.concatenate([pos, quat])
    

def compute_delta_action(state, action):
    """
    Compute the delta action between a given state and action while handling 2π wrap for orientation.
    
    Parameters:
        state  : tuple (x, y, z, roll, pitch, yaw)  -> Current state
        action : tuple (x, y, z, roll, pitch, yaw)  -> Target action
    
    Returns:
        delta_action : tuple (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)
    """
    # Compute linear deltas
    delta_x = action[0] - state[0]
    delta_y = action[1] - state[1]
    delta_z = action[2] - state[2]
    
    # Compute angular deltas with 2π wrap-around
    delta_roll = (action[3] - state[3] + np.pi) % (2 * np.pi) - np.pi
    delta_pitch = (action[4] - state[4] + np.pi) % (2 * np.pi) - np.pi
    delta_yaw = (action[5] - state[5] + np.pi) % (2 * np.pi) - np.pi
    
    return np.array([delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw])


def compute_action(state, delta_action):
    """
    Compute the new action given the initial state and delta action while handling 2π wrap for orientation.
    
    Parameters:
        state        : tuple (x, y, z, roll, pitch, yaw)  -> Initial state
        delta_action : tuple (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)  -> Delta action
    
    Returns:
        action : tuple (x, y, z, roll, pitch, yaw) -> Computed action
    """
    # Compute linear positions
    x_new = state[0] + delta_action[0]
    y_new = state[1] + delta_action[1]
    z_new = state[2] + delta_action[2]

    # Compute new orientations with 2π wrap-around
    roll_new = (state[3] + delta_action[3]) % (2 * np.pi)
    pitch_new = (state[4] + delta_action[4]) % (2 * np.pi)
    yaw_new = (state[5] + delta_action[5]) % (2 * np.pi)

    # Ensure angles remain in range [-π, π]
    if roll_new > np.pi:
        roll_new -= 2 * np.pi
    if pitch_new > np.pi:
        pitch_new -= 2 * np.pi
    if yaw_new > np.pi:
        yaw_new -= 2 * np.pi

    return np.array([x_new, y_new, z_new, roll_new, pitch_new, yaw_new])

def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class NeverStop(Condition):
    def condition_met(self):
        return False, False
    

class CustomRLBenchEnv(RLBenchEnv):

    def __init__(self,
                 task_class: Type[Task],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,
                 time_in_state: bool = False,
                 include_lang_goal_in_obs: bool = False,
                 record_every_n: int = 20):
        super(CustomRLBenchEnv, self).__init__(
            task_class, observation_config, action_mode, dataset_root,
            channels_last, headless=headless,
            include_lang_goal_in_obs=include_lang_goal_in_obs)
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            'IKError': 0,
            'ConfigurationPathError': 0,
            'InvalidActionError': 0,
        }
        self._last_exception = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                oe.shape = (oe.shape[0] - 7 * 3 + int(self._time_in_state),)  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (1. - ((self._i if t is None else t) / float(
                self._episode_length - 1))) * 2. - 1.
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        # obs_dict['gripper_pose'] = grip_pose
        return obs_dict

    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomRLBenchEnv, self).reset()
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts['InvalidActionError'] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail'),
                vid, fps=30))

            # error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i):
        self._i = 0
        # super(CustomRLBenchEnv, self).reset()

        self._task.set_variation(-1)
        d, = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)
        self._lang_goal = self._task.get_task_descriptions()[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict

    
class CustomRLRenchEnv2(CustomRLBenchEnv):
    def __init__(self, record_queue=None, never_terminal=False, unseen_task=True, *args, **kwargs):
        super(CustomRLRenchEnv2, self).__init__(*args, **kwargs)
        self._task_classes = [task_file_to_task_class(task) for task in RLBENCH_TASKS]
        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}
        self.record_queue = record_queue
        self.never_terminal = never_terminal
        self.unseen_task = unseen_task

    
    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = False
        return self._previous_obs_dict
    
    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.set_position([-0.7, 0.0, 0.2])
            cam_base.rotate([0, np.pi/12, 0])
            self._record_cam = VisionSensor.create([384, 384])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def _my_callback(self):
        if self._record_current_episode and self.record_queue is not None:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self.record_queue.put(self.normalize_image(cap))
    
    def normalize_image(self, img):
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return img
    
    def set_new_task(self, task_name: str):
        # Adapted from YARR/yarr/envs/rlbench_env.py MultiTaskRLBenchEnv class
        assert task_name in RLBENCH_TASKS, f"Task {task_name} not found in RLBENCH_TASKS"
        self.task_name = task_name
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)
        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
    
    def reset_to_demo(self, i, not_load_image=True):
        self._i = 0
        self._task.set_variation(-1)

        if self.unseen_task:
            # variation_count = self._task.variation_count()
            # self._task.set_variation(np.random.randint(variation_count))
            if self.task_name in ["close_drawer", "pick_up_cup", "reach_target"]:
                variation_count = self._task.variation_count()
                self._task.set_variation(i % variation_count)
            else:
                self._task.set_variation(0)
            with open(f"/home/daiyp/manipulation/RVT/rvt/gradio_demo/random_seeds/random_seed{i}.pkl", 'rb') as f:
                random_seed = pickle.load(f)
            np.random.set_state(random_seed)
            desc, obs = self._task.reset()
        else:
            d = self._task.get_demos(
                1, live_demos=False, 
                image_paths=not_load_image,  # not load image
                random_selection=False, from_episode_number=i)[0]
            self.d = d
            self._task.set_variation(d.variation_number)
            desc, obs = self._task.reset_to_demo(d)


        obs_copy = copy.deepcopy(obs)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = True if self.record_queue is not None else False
        self._episode_index += 1
        self._recorded_images.clear()
        if self.record_queue is not None:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self.record_queue.put(self.normalize_image(cap))
        # self.record_queue.put(self.normalize_image(self._previous_obs_dict['front_rgb']))

        return self._previous_obs_dict, obs_copy
    
    def step(self, act_result: ActResult) -> Transition:
        if self.never_terminal:
            self._task._task._success_conditions = [NeverStop()]
            
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.
        error_status = "success"
        info = {}

        try:
            obs, reward, terminal = self._task.step(action)
            if self.never_terminal:
                terminal = False
                reward = 0.0
            obs_copy = copy.deepcopy(obs)
            # obs_copy.gripper_pose = action[:7]
            # obs_copy.gripper_open = action[7]
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0
            obs_copy = None

            if isinstance(e, IKError):
                print("IKError")
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                print("ConfigurationPathError")
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                print("InvalidActionError")
                error_status = "error"
                self._error_type_counts['InvalidActionError'] += 1
            else:
                print("Unknown error")
            print(e)

            self._last_exception = e
        
        info.update({'error_status': error_status, 'obs': obs_copy})
        self._i += 1

        if self.record_queue is not None:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self.record_queue.put(self.normalize_image(cap))
        
        return Transition(obs, reward, terminal, info=info, summaries=[])
    






START_ACTION = np.array([0.2785,-0.0082,1.4719,-0.0,0.9927,-0.0,0.1209,1.0,0.0])

ROLLOUT_IMAGE_SIZE = 128


@dataclass
class Action:
    translation: ndarray    # (3,)
    rotation: quaternion.quaternion       # (4,) quaternion
    gripper_open: bool      
    ignore_collision: bool
    
    @property
    def T(self):
        return self.translation
    
    @T.setter
    def T(self, value):
        self.translation = value
    
    @property
    def R(self):
        return self.rotation
    
    @property
    def Rmat(self):
        return quaternion.as_rotation_matrix(self.rotation)
    
    @R.setter
    def R(self, value):
        self.rotation = value

    def to_numpy(self) -> ndarray:
        return np.concatenate((
            self.translation,
            self.quat_to_array(self.rotation, style='xyzw'),
            np.array([self.gripper_open, self.ignore_collision], dtype=float)
        ))
    
    @classmethod
    def from_numpy(cls, arr: ndarray):
        translation = arr[:3]
        rotation = cls.array_to_quat(arr[3:7], style='xyzw')
        gripper_open = bool(arr[7])
        ignore_collision = bool(arr[8])
        return cls(translation, rotation, gripper_open, ignore_collision)
    
    @staticmethod
    def quat_to_array(quat: quaternion.quaternion, style: str = 'xyzw'):
        # style is the output arr style
        a = quaternion.as_float_array(quat)
        if style == 'xyzw':
            return np.array([a[1], a[2], a[3], a[0]])
        elif style == 'wxyz':
            return a
        else:
            raise ValueError(f"Unknown style: {style}")
    
    @staticmethod
    def array_to_quat(arr: ndarray, style: str = 'xyzw'):
        # style is the input arr style
        if style == 'xyzw':
            return quaternion.quaternion(arr[3], arr[0], arr[1], arr[2])
        elif style == 'wxyz':
            return quaternion.quaternion(arr[0], arr[1], arr[2], arr[3])
        else:
            raise ValueError(f"Unknown style: {style}")
        
    @staticmethod
    def quat_to_euler(quat: quaternion.quaternion):
        return quaternion.as_euler_angles(quat)
    
    @staticmethod
    def delta_action(action_from: 'Action', action_to: 'Action'):
        delta_translation = action_to.T - action_from.T
        delta_rotation = Rotation.from_quat(action_from.quat_to_array(action_from.R.inverse() * action_to.R, 'xyzw')).as_euler('xyz', degrees=True)
        delta_gripper = int(action_to.gripper_open) - int(action_from.gripper_open)
        delta_ignore = int(action_to.ignore_collision) - int(action_from.ignore_collision)
        return {
            "translation": delta_translation,  
            "rotation": delta_rotation, 
            "gripper": delta_gripper,   # 0 = unchanged gripper state, 1 = open gripper, -1 = close gripper
            "collision": delta_ignore   # 0 = unchanged collision state, 1 = ignore collision, -1 = consider collision
        }

    def __str__(self):
        return_str = f"T: {self.T}\t"
        return_str += f"R: {self.quat_to_array(self.R, style='xyzw')}"
        if self.gripper_open:
            return_str += " gripper open "
        else:
            return_str += " gripper close"
        if self.ignore_collision:
            return_str += ", collision ignore"
        else:
            return_str += ", collision consider"
        return return_str



TEMP_0_nolabel = "<image>\nThe task goal is: {task_goal}. This is the first step and the robot is about to start the task. Based on the visual observation and the context, what's the next instruction for the robot arm?"
TEMP_nolabel = "<image>\nThe task goal is: {task_goal}. In the previous step, the robot arm was given the following instruction: \"{previous_instruction}\". {robot_delta_state} Based on the visual observation and the context, what's the next instruction for the robot arm?"
TEMP_0_label = "<image>\nThe task goal is: {task_goal}. This is the first step and the robot is about to start the task. Based on the visual observation and the context, how does the robot fulfil that previous instruction and what's the next instruction for the robot arm?"
TEMP_label = "<image>\nThe task goal is: {task_goal}. In the previous step, the robot arm was given the following instruction: \"{previous_instruction}\". {robot_delta_state} Based on the visual observation and the context, how does the robot fulfil that previous instruction and what's the next instruction for the robot arm?"
    
AXES = ["x", "y", "z"]
TRANSLATION_SMALL_THRES = 0.01
TRANSLATION_LARGE_THRES = 0.05
ROTATION_SMALL_THRES = 5
ROTATION_LARGE_THRES = 20
# Directions: (name, axis, +ve sign = 1)
# backward = closer to VLM's view & forward = further away from VLM's perspective
DIRECTIONS = [("backward", 0, 1), ("forward", 0, -1), ("right", 1, 1), ("left", 1, -1), ("down", 2, -1), ("up", 2, 1)]
