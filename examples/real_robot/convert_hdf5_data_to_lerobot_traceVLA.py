from pathlib import Path
import shutil
import sys
import time

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro
from scipy.spatial.transform import Rotation as R
import h5py

np.set_printoptions(precision=5, suppress=True)
from PIL import Image, ImageDraw
import os
from gradio_client import Client
from typing import Literal
import base64
import io

# Show Dataset Structure (Recursively)
def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")


def img_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def base64_to_img(base64_str: str):
    img_str = base64.b64decode(base64_str.split("base64,")[1])
    return Image.open(io.BytesIO(img_str))

class TraceClient:
    
    def __init__(self):
        gradio_public_url = "https://4bff1c13e7dbea04f9.gradio.live"
        self.client = Client(gradio_public_url)
        self.client.view_api()
   
    def query_trace(
        self,
        image: str | Image.Image,
        task_description: str,
        which_shoulder: Literal["left", "right"],
        reset_trace: bool = False,
    ) -> dict:
        # Check if image exists
        if isinstance(image, str):
            if os.path.exists(image):
                image_input = {"path": image}
            elif image.startswith("data:image"):
                image_input = {"url": image}
            else:
                raise ValueError(f"Invalid image input: {image}")
        elif isinstance(image, Image.Image):
            # convert PIL Image to base64 string
            img_str = img_to_base64(image)
            image_input = {"url": f"data:image/jpeg;base64,{img_str}"}
        else:
            raise ValueError(f"Invalid image input: {image}")

        # Submit the prediction
        try:
            action_str, trace_img_file, pred_time = self.client.predict(
                image=image_input,
                task_description=task_description,
                unnorm_key="bc_z",
                which_hand=which_shoulder,
                reset_trace=reset_trace # reset trace tracker at the beginning of each episode
            )
            # print("pred_time:", pred_time)
            if trace_img_file is not None:
                # trace_img_file is a temporary file created by gradio
                trace_img_file = Image.open(trace_img_file)
            return trace_img_file
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None



def is_noop(action, prev_action=None, threshold=1e-3, is_cartesian=False):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return True
    
    if not is_cartesian:
        delta_action = action - prev_action
        # print("delta joint position action:", delta_action, np.linalg.norm(delta_action[:-1]))
    else:
        prev_action_rotmax = R.from_euler("xyz", prev_action[3:6], degrees=False)
        prev_action_trans = prev_action[:3]
        
        action_rotmax = R.from_euler("xyz", action[3:6], degrees=False)
        action_trans = action[:3]
        
        delta_action_rotmax = action_rotmax * prev_action_rotmax.inv()
        delta_action_trans = action_trans - prev_action_trans
        
        delta_action = np.concatenate([delta_action_trans, delta_action_rotmax.as_euler("xyz", degrees=False)])
        delta_action = np.concatenate([delta_action, [action[-1]-prev_action[-1]]])
        # print("delta cartesian action:", delta_action, np.linalg.norm(delta_action[:-1]))
        
    
    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(delta_action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(data_dir: str = "/data/daiyp/crosshair/real_data", repo_id: str = "realrobot_tracevlaxxxx", use_reticle: bool = False, use_robotpoint: bool = False, use_trace: bool = True):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)
        
    if use_reticle:
        print("Using reticle images")

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=15,
        features={
            "left_shoulder_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "right_shoulder_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["joint_states"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["joint_actions"],
            },
            # "cartesian_states": {
            #     "dtype": "float32",
            #     "shape": (7,),
            #     "names": ["cartesian_states"],
            # },
            # "cartesian_actions": {
            #     "dtype": "float32",
            #     "shape": (7,),
            #     "names": ["cartesian_actions"],
            # },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # find all hdf5 files in the data_dir, recursively
    hdf5_files = list(Path(data_dir).glob("**/*.hdf5")) 
    exclude_dirs = ["tennis_ball_in_bowl", "play", "v2"]
    hdf5_files = [file for file in hdf5_files if not any(exclude_dir in str(file) for exclude_dir in exclude_dirs)]
    print(f"Found {len(hdf5_files)} hdf5 files")

    trace_client = TraceClient()

    
    for hdf5_file in tqdm.tqdm(hdf5_files):  
        data = h5py.File(hdf5_file, 'r')
        language_instruction = hdf5_file.name.split("-")[0].replace("_", " ")
        print(language_instruction)
        
        action_joint_position = data["action/joint_position"]
        action_gripper_position = data["action/gripper_position"]
        action_cartesian_position = data["action/cartesian_position"]
        
        state_joint_position = data["observation/robot_state/joint_positions"]
        state_gripper_position = data["observation/robot_state/gripper_position"]
        state_cartesian_position = data["observation/robot_state/cartesian_position"]
        
        prev_joint_actions = None
        prev_cartesian_actions = None
        
        
        if use_reticle:
            left_shoulder_image = data["observation/camera_rgb_reticle/left_shoulder"]
            right_shoulder_image = data["observation/camera_rgb_reticle/right_shoulder"]
            wrist_image = data["observation/camera_rgb_reticle/wrist"]
        else:
            left_shoulder_image = data["observation/camera_rgb/left_shoulder"]
            right_shoulder_image = data["observation/camera_rgb/right_shoulder"]
            wrist_image = data["observation/camera_rgb/wrist"]

        length = len(action_joint_position)
                
        record = False
        reset_trace = True
        for idx in range(length):
            joint_states =  np.concatenate([state_joint_position[idx], [state_gripper_position[idx]]])
            joint_actions = np.concatenate([action_joint_position[idx], [action_gripper_position[idx]]])
            
            cartesian_states = np.concatenate([state_cartesian_position[idx], [state_gripper_position[idx]]])
            cartesian_actions = np.concatenate([action_cartesian_position[idx], [action_gripper_position[idx]]])
        
            
            # print("joint_actions:", joint_actions)
            # print("joint_states:", joint_states)
            
            # print("cartesian_actions:", cartesian_actions)
            # print("cartesian_states:", cartesian_states)
            
            if idx>0 and (is_noop(joint_actions, prev_joint_actions, threshold=1e-3) or \
                is_noop(cartesian_actions, prev_cartesian_actions, threshold=1e-4, is_cartesian=True)):
                # print("noop, skipping")
                continue
            
            
            if use_trace:    
                try:
                    traced_left_shoulder_img = trace_client.query_trace(
                        image=Image.fromarray(left_shoulder_image[idx]),
                        task_description=language_instruction,
                        which_shoulder="left",
                        reset_trace=reset_trace,
                    )
                    
                    traced_right_shoulder_img = trace_client.query_trace(
                        image=Image.fromarray(right_shoulder_image[idx]),
                        task_description=language_instruction,
                        which_shoulder="right",
                        reset_trace=reset_trace,
                    )
                    reset_trace = False
                    if traced_left_shoulder_img is None:
                        traced_left_shoulder_img = Image.fromarray(left_shoulder_image[idx])
                    if traced_right_shoulder_img is None:
                        traced_right_shoulder_img = Image.fromarray(right_shoulder_image[idx])

                    
                    # Image -> numpy array
                    left_shoulder_img = np.array(traced_left_shoulder_img)
                    right_shoulder_img = np.array(traced_right_shoulder_img)
                    wrist_img = np.array(wrist_image[idx])
               
                    traced_left_shoulder_img.save(f"sandbox/traced_left_shoulder_image.png")
                    traced_right_shoulder_img.save(f"sandbox/traced_right_shoulder_image.png")
                
                except Exception as e:
                    print(f"Error querying tracevla: {e}")
                    left_shoulder_img = left_shoulder_image[idx]
                    right_shoulder_img = right_shoulder_image[idx]
                    wrist_img = wrist_image[idx]
                                 
            
            # if use_trace:
            #     pass
            else:
                left_shoulder_img = left_shoulder_image[idx]
                right_shoulder_img = right_shoulder_image[idx]
                wrist_img = wrist_image[idx]
            
            
            dataset.add_frame(
                {
                    "left_shoulder_image": left_shoulder_img,
                    "right_shoulder_image": right_shoulder_img,
                    "wrist_image": wrist_img,
                    "state": joint_states.astype(np.float32),
                    "actions": joint_actions.astype(np.float32),
                    # "cartesian_states": cartesian_states.astype(np.float32),
                    # "cartesian_actions": cartesian_actions.astype(np.float32),
                }
            )
            record = True
            
            prev_joint_actions = joint_actions
            prev_cartesian_actions = cartesian_actions
            
            # Image.fromarray(left_shoulder_image[idx]).save(f"sandbox/left_shoulder_image.png")
            # Image.fromarray(right_shoulder_image[idx]).save(f"sandbox/right_shoulder_image.png")
            # Image.fromarray(wrist_image[idx]).save(f"sandbox/wrist_image.png")
            input("...")

        if record:
            dataset.save_episode(task=language_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

if __name__ == "__main__":
    tyro.cli(main)
