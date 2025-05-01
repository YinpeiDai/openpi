

from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R
import h5py
import re

from PIL import Image

np.set_printoptions(precision=5, suppress=True)

# Show Dataset Structure (Recursively)
def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")



def is_open(gripper_qpos):
    if abs(gripper_qpos[0]) > 0.035 and abs(gripper_qpos[1]) > 0.035:
        return True
    return False

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


def main():
    # Clean up any existing dataset in the output directory
    repo_id = "realrobot_pretrain_data_xxx"
    # output_path = LEROBOT_HOME / repo_id
    LEROBOT_HOME = Path("/home/daiyp/openpi/lerobot_d_real")
    output_path = Path("/home/daiyp/openpi/lerobot_d_real") / repo_id
    
    if output_path.exists():
        shutil.rmtree(output_path)

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
    
    # for hdf5_file in Path("/data/daiyp/crosshair/real_data/play").glob("*.hdf5"):  
    #     data = h5py.File(hdf5_file, 'r')
    #     language_instruction = hdf5_file.name.split("-")[0].replace("_", " ")
    #     print(language_instruction)
        
    #     action_joint_position = data["action/joint_position"]
    #     action_gripper_position = data["action/gripper_position"]
    #     action_cartesian_position = data["action/cartesian_position"]
        
    #     state_joint_position = data["observation/robot_state/joint_positions"]
    #     state_gripper_position = data["observation/robot_state/gripper_position"]
    #     state_cartesian_position = data["observation/robot_state/cartesian_position"]
        
    #     prev_joint_actions = None
    #     prev_cartesian_actions = None
        
    
    #     left_shoulder_image = data["observation/camera_rgb_reticle/left_shoulder"]
    #     right_shoulder_image = data["observation/camera_rgb_reticle/right_shoulder"]
    #     wrist_image = data["observation/camera_rgb_reticle/wrist"]

    #     length = len(action_joint_position)
        
    #     for idx in range(length):
    #         joint_states =  np.concatenate([state_joint_position[idx], [state_gripper_position[idx]]])
    #         joint_actions = np.concatenate([action_joint_position[idx], [action_gripper_position[idx]]])
            
    #         cartesian_states = np.concatenate([state_cartesian_position[idx], [state_gripper_position[idx]]])
    #         cartesian_actions = np.concatenate([action_cartesian_position[idx], [action_gripper_position[idx]]])
            
    #         if idx>0 and (is_noop(joint_actions, prev_joint_actions, threshold=1e-3) or is_noop(cartesian_actions, prev_cartesian_actions, threshold=1e-4, is_cartesian=True)):
    #             print("noop, skipping")
    #             continue
            
    #         dataset.add_frame(
    #             {
    #                 "left_shoulder_image": left_shoulder_image[idx],
    #                 "right_shoulder_image": right_shoulder_image[idx],
    #                 "wrist_image": wrist_image[idx],
    #                 "state": joint_states.astype(np.float32),
    #                 "actions": joint_actions.astype(np.float32),
    #             }
    #         )
    #         # print("joint_states:", joint_states)
    #         # print("joint_actions:", joint_actions)
    #         # Image.fromarray(left_shoulder_image[idx]).save(f"left_shoulder_image.png")
    #         # Image.fromarray(right_shoulder_image[idx]).save(f"right_shoulder_image.png")
    #         # Image.fromarray(wrist_image[idx]).save(f"wrist_image.png")
    #         # input("...")
        
    #         prev_joint_actions = joint_actions
    #         prev_cartesian_actions = cartesian_actions
                   
    #     dataset.save_episode(task=language_instruction)
    
    
    
    for hdf5_file in Path("/home/daiyp/openvla/data_regenerated_hdf5_libero/droid").glob("*/*.hdf5"):  
        print(hdf5_file)
        data = h5py.File(hdf5_file, 'r')
        # extract task name
        words = hdf5_file.name[:-10].split("_")
        command = ''
        for w in words:
            if "SCENE" in w:
                command = ''
                continue
            command = command + w + ' '
        language_instruction = command[:-1]
        print(language_instruction)
        
        recorded = False
        
        for k in data["data"].keys():
            demo = data["data"][k]
            gripper_states = demo["obs"]["gripper_states"]
            joint_states = demo["obs"]["joint_states"]
            joint_actions = demo["obs"]["joint_actions"]
            actions = demo["obs"]["actions"]
            left_shoulder_image = demo["obs"]["left_shoulder_rgb"]
            right_shoulder_image = demo["obs"]["right_shoulder_rgb"]
            wrist_image = demo["obs"]["wrist_rgb"]
            
            assert len(gripper_states) == len(joint_states) == len(joint_actions) == len(actions)
            
            recorded = False
            
            for i in range(len(gripper_states)):
                if actions[i][-1] < 0:
                    openness = 0
                else:
                    openness = 1
                
                # print("action:", actions[i])
                if i+1 < len(gripper_states):
                    joint_action = np.concatenate([joint_actions[i+1], [openness]])
                else:
                    joint_action = np.concatenate([joint_actions[i], [openness]])
                joint_state = np.concatenate([joint_states[i], [1-abs(gripper_states[i][0])/0.04]])
                
                # print("joint_state:", joint_state)
                # print("joint_action:", joint_action)
                # Image.fromarray(left_shoulder_image[i]).save(f"left_shoulder_image.png")
                # Image.fromarray(right_shoulder_image[i]).save(f"right_shoulder_image.png")
                # Image.fromarray(wrist_image[i]).save(f"wrist_image.png")
                
                # input("...")
                
                dataset.add_frame(
                    {
                        "left_shoulder_image": left_shoulder_image[i],
                        "right_shoulder_image": right_shoulder_image[i],
                        "wrist_image": wrist_image[i],
                        "state": joint_state,
                        "actions": joint_action,
                    }
                )
                recorded = True
            
            if recorded:
                dataset.save_episode(task=language_instruction)
        
    
    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

if __name__ == "__main__":
    tyro.cli(main)
