import collections
from copy import deepcopy
import dataclasses
import json
import logging
import math
import os
import pathlib
import re

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro



from copy import deepcopy
import math
import os
import pickle

import imageio

import h5py
import numpy as np


import numpy as np

from PIL import Image
import json


from crosshair.reticle_builder import ReticleBuilder
from crosshair.video_recorder import VideoRecorder
from crosshair.config import CONFIG_DICT




from simulator import RLBenchSim
from utils import quat2axisangle, rpy2quat, quat2rpy, compute_delta_action, compute_action


RLBENCH_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RLBENCH_ENV_RESOLUTION = 256  # resolution used to render training data

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

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_name: str = "close_jar" # all

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_path: str = "/home/daiyp/openpi/runs/evaluation/rlbench"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    
    model_name: str = "pi0_fast_libero"             # Model name
    
    save_video_num: int = 25  # Number of videos to save per task
    
    use_reticle: bool = False  # Use reticle in the environment
    reticle_config_key: str = "large_crosshair_dynamic_default_color"  # Reticle configuration key

def extract(transition):
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
            

def eval_rlbench(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)    
    env = RLBenchSim(
        task_name="open_drawer",
        dataset_root="/nfs/turbo/coe-chaijy-unreplicated/datasets/rlbench/test",
        use_depth=True,
        resolution=RLBENCH_ENV_RESOLUTION
    )

    if task_name == "all":
        task_name_list = RLBENCH_TASKS
    elif "," in task_name:
        task_name_list = task_name.split(",")
    else:
        task_name_list = [task_name]
    
    for task_name in task_name_list:
        save_dir = os.path.join(args.save_path, task_name, args.model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_dir_video = os.path.join(save_dir, "video")
        os.makedirs(save_dir_video, exist_ok=True)
        
        task_name = task_name
        if task_name == "close_jar":
            max_steps = 270
        elif task_name == "insert_onto_square_peg":
            max_steps = 320
        elif task_name == "light_bulb_in":
            max_steps = 400
        elif task_name == "meat_off_grill":
            max_steps = 260
        elif task_name == "open_drawer":
            max_steps = 150
        elif task_name == "push_buttons":
            max_steps = 220
        elif task_name == "stack_blocks":
            max_steps = 900
        elif task_name == "stack_cups":
            max_steps = 360
        else:
            raise ValueError(f"Unknown task: {task_name}")


        print(f"connected to {args.host}:{args.port}")
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

        print(f"Eval task name: {task_name }")        
        
        if args.use_reticle:
            print(f"Using reticle with configuration key: {args.reticle_config_key}")
            config = CONFIG_DICT[args.reticle_config_key]
            shooting_line_config = config["shooting_line"]
            scope_reticle_config = config["scope_reticle"]
            MAX_EE_TABLE_DIST = 0.4
            FIXCAM_TOLERANCE = 18
            WSTCAM_TOLERANCE = 12
            scope_reticle_config.line_length_cfg.maxdist = MAX_EE_TABLE_DIST
            reticle_builder = ReticleBuilder(
                shooting_line_config=shooting_line_config,
                scope_reticle_config=scope_reticle_config,
            )

        results = []
        
        # Start episodes 
        for episode_num in tqdm.tqdm(range(25)):
            env.set_new_task(task_name)
            _, obs = env.reset(episode_num)
            
            pkl_file = os.path.join(env.dataset_root, task_name, "all_variations", "episodes", f"episode{episode_num}", "variation_descriptions.pkl")
            with open(pkl_file, "rb") as f:
                task_goals = pickle.load(f)
            task_goal = task_goals[0]
            task_description = task_goal
            
            print(f"Task: {task_name}, Episode: {episode_num}, Goal: {task_description}")
            
            if "blocks" in task_goal:
                num_blocks = int(re.findall(r"\d+", task_goal)[0])
                if num_blocks == 2:
                    max_steps = 550
                elif num_blocks == 3:
                    max_steps = 700
                elif num_blocks == 4:
                    max_steps = 900
        
            success = False
            for _ in range(10):
                action = np.concatenate([obs.gripper_pose, [1.0], [1.0]])
                transition = env.step(action)

            # Setup
            t = 0
            replay_images = []
            action_plan = collections.deque()

            while t < max_steps:
                try:    
                    rlbench_state, robot_state, gripper_state, joint_state, ee_state, gripper_open, \
                        front_rgb, front_depth, front_camera_extrinsics, front_camera_intrinsics, \
                        wrist_rgb, wrist_depth, wrist_camera_extrinsics, wrist_camera_intrinsics = extract(transition)
                    
                    # Image.fromarray(front_rgb).save(f"front_rgb.png")
                    # Image.fromarray(wrist_rgb).save(f"wrist_rgb.png")
                    # input("Press Enter to continue...")
                    
                    if args.use_reticle:
                        front_rgb_reticle = deepcopy(front_rgb)
                        front_rgb_reticle = reticle_builder.render_on_fix_camera(
                            camera_rgb=front_rgb_reticle,
                            camera_depth=front_depth,
                            camera_extrinsics=np.linalg.inv(front_camera_extrinsics),
                            camera_intrinsics=front_camera_intrinsics,
                            gripper_pos=rlbench_state[:3],
                            gripper_quat=rlbench_state[3:7],
                            gripper_open=gripper_open,
                            image_height=RLBENCH_ENV_RESOLUTION,
                            image_width=RLBENCH_ENV_RESOLUTION,
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
                            image_height=RLBENCH_ENV_RESOLUTION,
                            image_width=RLBENCH_ENV_RESOLUTION,
                            tolerance=WSTCAM_TOLERANCE
                        )
                        img = np.ascontiguousarray(front_rgb_reticle)
                        wrist_img = np.ascontiguousarray(wrist_rgb_reticle)
                    else:
                        img = np.ascontiguousarray(front_rgb)
                        wrist_img = np.ascontiguousarray(wrist_rgb)

                    
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(np.concatenate((img, wrist_img), axis=1))

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((ee_state, gripper_state), axis=-1),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    libero_action = action_plan.popleft()
                    # print(f"Action: {libero_action}")

                    # Execute action in environment
                    libero_state = quat2rpy(rlbench_state)
                    recover_gripper_pose = compute_action(state=libero_state, delta_action=libero_action[:6])
                    recover_rlbench_gripper_pose = rpy2quat(recover_gripper_pose)
                    recoevr_rlbench_gripper_open = 1.0 if libero_action[6] < 0 else 0.0
                    recover_rlbench_action = np.concatenate([recover_rlbench_gripper_pose, [recoevr_rlbench_gripper_open], [1.0]])
            
                    
                    transition = env.step(recover_rlbench_action)
                    if transition.terminal and transition.reward == 100.0:
                        success =True
                        break
                    else:
                        success = False

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            # Save a replay video of the episode
            suffix = "success" if success else "failure"
            task_segment = task_description.replace(" ", "_")
            
            if episode_num < args.save_video_num:
                imageio.mimwrite(
                    os.path.join(save_dir, "video", f"task{episode_num}-seed{args.seed}-{task_segment}_ep{episode_num}_{suffix}.mp4"),
                    [np.asarray(x) for x in replay_images],
                    fps=30,
                )
                
            results.append({"episode": episode_num, "success": success, "task_description": task_description})


        processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
        json_name = f"task{episode_num}-seed{args.seed}-{processed_task_description}.json"
        with open(os.path.join(save_dir, json_name), "w") as f:
            json.dump(results, f, indent=2)

    env.close()


if __name__ == "__main__":
    eval_rlbench(tyro.cli(Args))
