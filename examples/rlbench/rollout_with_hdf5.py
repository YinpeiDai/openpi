

from copy import deepcopy
import math
import os
import pickle

import imageio

import h5py
import numpy as np
from tqdm import tqdm

import numpy as np

from PIL import Image
import json


from crosshair.reticle_builder import ReticleBuilder
from crosshair.video_recorder import VideoRecorder
from crosshair.config import CONFIG_DICT


# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simulator import RLBenchSim
from utils import quat2axisangle, rpy2quat, quat2rpy, compute_delta_action, compute_action

RLBENCH_TASKS = [
    "open_drawer",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "light_bulb_in",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]

IMAGE_RESOLUTION = 256




class Rollout:
    def __init__(self):
        self.env = RLBenchSim(
            task_name="open_drawer",
            dataset_root="/nfs/turbo/coe-chaijy-unreplicated/datasets/rlbench/train",
            use_depth=True,
            resolution=IMAGE_RESOLUTION
        )
        
        self.target_dir = "/home/daiyp/openpi/runs/evaluation/rollout_hdf5"
        os.makedirs(os.path.join(self.target_dir, "video"), exist_ok=True)
        self.all_views = ["front", "wrist"]
                    
    @staticmethod
    def normalize(action):
        action[3:7] = action[3:7] / np.linalg.norm(action[3:7])
        return action
    
    
    def extract(self, transition):
        obs = transition.info["obs"]
        gripper_pose = obs.gripper_pose
        gripper_qpos = obs.gripper_joint_positions
        gripper_open = obs.gripper_open
        joint_position = obs.joint_positions
                
        front_rgb = obs.front_rgb
        front_depth = obs.front_depth.squeeze()
        front_camera_extrinsics = obs.misc["front_camera_extrinsics"]
        front_camera_intrinsics = obs.misc["front_camera_intrinsics"]
        
        wrist_rgb = obs.wrist_rgb
        wrist_depth = obs.wrist_depth.squeeze()
        wrist_camera_extrinsics = obs.misc["wrist_camera_extrinsics"]
        wrist_camera_intrinsics = obs.misc["wrist_camera_intrinsics"]
                
        
        robot_state = np.concatenate([gripper_qpos, gripper_pose])
        gripper_state = gripper_qpos
        joint_state = joint_position
        ee_state = np.hstack((gripper_pose[:3], quat2axisangle(gripper_pose[3:])))
        
        return gripper_pose, robot_state, gripper_state, joint_state, ee_state, gripper_open, \
            front_rgb, front_depth, front_camera_extrinsics, front_camera_intrinsics, \
            wrist_rgb, wrist_depth, wrist_camera_extrinsics, wrist_camera_intrinsics
                
        
    
    def eval_episode(self, task_name, episode_num, actions, config_key="large_crosshair_dynamic_default_color"):
        _, obs = self.env.reset(episode_num)
        
        pkl_file = os.path.join(self.env.dataset_root, task_name, "all_variations", "episodes", f"episode{episode_num}", "variation_descriptions.pkl")
        with open(pkl_file, "rb") as f:
            task_goals = pickle.load(f)
        task_goal = task_goals[0]

        
        for _ in range(10):
            action = np.concatenate([obs.gripper_pose, [1.0], [1.0]])
            transition = self.env.step(action)
            
            
        
       
                
        config = CONFIG_DICT[config_key]
        shooting_line_config = config["shooting_line"]
        scope_reticle_config = config["scope_reticle"]
        MAX_EE_TABLE_DIST = 0.7
        FIXCAM_TOLERANCE = 12
        WSTCAM_TOLERANCE = 8
        
        scope_reticle_config.line_length_cfg.maxdist = MAX_EE_TABLE_DIST

        reticle_builder = ReticleBuilder(
            shooting_line_config=shooting_line_config,
            scope_reticle_config=scope_reticle_config,
        )
        
        
        replay_images = []
        
        for libero_action in actions:
            
            rlbench_state, robot_state, _, joint_state, ee_state, gripper_open, \
            front_rgb, front_depth, front_camera_extrinsics, front_camera_intrinsics, \
            wrist_rgb, wrist_depth, wrist_camera_extrinsics, wrist_camera_intrinsics = self.extract(transition)
            
            
            ##---
            front_rgb_reticle = deepcopy(front_rgb)
            front_rgb_reticle = reticle_builder.render_on_fix_camera(
                camera_rgb=front_rgb_reticle,
                camera_depth=front_depth,
                camera_extrinsics=np.linalg.inv(front_camera_extrinsics),
                camera_intrinsics=front_camera_intrinsics,
                gripper_pos=rlbench_state[:3],
                gripper_quat=rlbench_state[3:7],
                gripper_open=gripper_open,
                image_height=IMAGE_RESOLUTION,
                image_width=IMAGE_RESOLUTION,
                tolerance=FIXCAM_TOLERANCE,
            )
            
            wrist_rgb_reticle = deepcopy(wrist_rgb)
            wrist_rgb_reticle = reticle_builder.render_on_wst_camera(
                wrist_camera_rgb=wrist_rgb_reticle,
                wrist_camera_depth=wrist_depth,
                wrist_camera_extrinsics=np.linalg.inv(wrist_camera_extrinsics),
                wrist_camera_intrinsics=wrist_camera_intrinsics,
                gripper_pos=rlbench_state[:3],
                gripper_quat=rlbench_state[3:7],
                gripper_open=gripper_open,
                image_height=IMAGE_RESOLUTION,
                image_width=IMAGE_RESOLUTION,
                tolerance=WSTCAM_TOLERANCE
            )
            ##---

            # recover, given libero_action
            # rlbench_state (7),  predicted libero_action (6) -> rlbench action (9)
            libero_state = quat2rpy(rlbench_state)
            recover_gripper_pose = compute_action(state=libero_state, delta_action=libero_action[:6])
            recover_rlbench_gripper_pose = rpy2quat(recover_gripper_pose)
            recoevr_rlbench_gripper_open = 1.0 if libero_action[6] < 0 else 0.0
            recover_rlbench_action = np.concatenate([recover_rlbench_gripper_pose, [recoevr_rlbench_gripper_open], [1.0]])
    
            
            transition = self.env.step(recover_rlbench_action)
            
            
            Image.fromarray(front_rgb_reticle).save(f"front_image.png")
            Image.fromarray(wrist_rgb_reticle).save(f"wrist_image.png")
            input("Press Enter to continue...")
            
            
        
        if transition.terminal and transition.reward == 100.0:
            success =True
        else:
            success = False
            
        imageio.mimwrite(
            os.path.join(self.target_dir, "video", f"{task_goal}_ep{episode_num}_{success}.mp4"),
            [np.asarray(x) for x in replay_images],
            fps=45,
        )
        
        return success
 


            
def main(hdf5_path):
    rollout = Rollout()
    
    hdf5_file = h5py.File(hdf5_path, "r")
    task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
    rollout.env.set_new_task(task_name)    
    data = hdf5_file["data"]
    for ep in data:
        print(f"Episode:{ep}")
        ep_num = int(ep.split("_")[-1])
        demo_data = data[ep]
        actions = demo_data["actions"][()]
        rollout.eval_episode(task_name, ep_num, actions)
    

if __name__ == "__main__":
    hdf5_path = "/home/daiyp/openpi/rlbench_regenerate_data/open_drawer_demo.hdf5"
    main(hdf5_path)


