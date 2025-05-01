

from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R
import h5py

np.set_printoptions(precision=5, suppress=True)

# Show Dataset Structure (Recursively)
def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")

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


def main(data_dir: str = "/home/ubuntu/chailab/daiyp/real_robot_data_h5/traced_h5dy", repo_id: str = "realrobot_tracevla"):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / repo_id
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
    
    for hdf5_file in Path(data_dir).glob("*.hdf5"):  
        data = h5py.File(hdf5_file, 'r')
        
        
        language_instruction = hdf5_file.name.split("-")[0].replace("_", " ")
        print(language_instruction)
        
        action_joint_position = data["action/joint_position"]
        action_gripper_position = data["action/gripper_position"]
        # action_cartesian_position = data["action/cartesian_position"]
        
        state_joint_position = data["observation/robot_state/joint_positions"]
        state_gripper_position = data["observation/robot_state/gripper_position"]
        # state_cartesian_position = data["observation/robot_state/cartesian_position"]
        
        prev_joint_actions = None
        prev_cartesian_actions = None
        
        left_shoulder_image = data["observation/camera_rgb/left_shoulder_trace"]
        right_shoulder_image = data["observation/camera_rgb/right_shoulder_trace"]
        wrist_image = data["observation/camera_rgb/wrist_trace"]

        length = len(action_joint_position)
        
        for idx in range(length):
            joint_states =  np.concatenate([state_joint_position[idx], [state_gripper_position[idx]]])
            joint_actions = np.concatenate([action_joint_position[idx], [action_gripper_position[idx]]])
            
            # cartesian_states = np.concatenate([state_cartesian_position[idx], [state_gripper_position[idx]]])
            # cartesian_actions = np.concatenate([action_cartesian_position[idx], [action_gripper_position[idx]]])
        
            
            # print("joint_actions:", joint_actions)
            # print("joint_states:", joint_states)
            
            # print("cartesian_actions:", cartesian_actions)
            # print("cartesian_states:", cartesian_states)
            
            if idx>0 and (is_noop(joint_actions, prev_joint_actions, threshold=1.2e-3)):
                print("noop, skipping")
                continue
            
            dataset.add_frame(
                {
                    "left_shoulder_image": left_shoulder_image[idx],
                    "right_shoulder_image": right_shoulder_image[idx],
                    "wrist_image": wrist_image[idx],
                    "state": joint_states.astype(np.float32),
                    "actions": joint_actions.astype(np.float32),
                    # "cartesian_states": cartesian_states.astype(np.float32),
                    # "cartesian_actions": cartesian_actions.astype(np.float32),
                }
            )
            
            prev_joint_actions = joint_actions
            # prev_cartesian_actions = cartesian_actions
            
            # from PIL import Image
            # Image.fromarray(left_shoulder_image[idx]).save(f"sandbox/left_shoulder_image_{idx}.png")
            # Image.fromarray(right_shoulder_image[idx]).save(f"sandbox/right_shoulder_image_{idx}.png")
            # Image.fromarray(wrist_image[idx]).save(f"sandbox/wrist_image_{idx}.png")
            # print("state:", joint_states)
            # print("action:", joint_actions)
            # input("...")
            
            
        dataset.save_episode(task=language_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

if __name__ == "__main__":
    tyro.cli(main)
