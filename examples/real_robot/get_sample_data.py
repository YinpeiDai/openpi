import h5py
from pathlib import Path
import numpy as np

use_reticle = True
data_dir = "/data/daiyp/crosshair/real_data/ball_in_drawer/"

folder_name = "reticle_samples_drawer"

for hdf5_file in Path(data_dir).glob("*.hdf5"):                
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
    
    
    for idx in range(length):
        if idx not in [0, length//3, 2*length//3]:
            continue
        
        joint_states =  np.concatenate([state_joint_position[idx], [state_gripper_position[idx]]])
        joint_actions = np.concatenate([action_joint_position[idx], [action_gripper_position[idx]]])
        
        cartesian_states = np.concatenate([state_cartesian_position[idx], [state_gripper_position[idx]]])
        cartesian_actions = np.concatenate([action_cartesian_position[idx], [action_gripper_position[idx]]])
    
        
        print("joint_actions:", joint_actions)
        print("joint_states:", joint_states)
        
        # print("cartesian_actions:", cartesian_actions)
        # print("cartesian_states:", cartesian_states)
        
        prev_joint_actions = joint_actions
        prev_cartesian_actions = cartesian_actions
        
        
        from PIL import Image
        Image.fromarray(left_shoulder_image[idx]).save(f"/home/daiyp/openpi/examples/real_robot/{folder_name}/left_shoulder_image_{idx}.png")
        Image.fromarray(right_shoulder_image[idx]).save(f"/home/daiyp/openpi/examples/real_robot/{folder_name}/right_shoulder_image_{idx}.png")
        Image.fromarray(wrist_image[idx]).save(f"/home/daiyp/openpi/examples/real_robot/{folder_name}/wrist_image_{idx}.png")
        
        # store joint_states and joint_actions into a text file
        with open(f"/home/daiyp/openpi/examples/real_robot/{folder_name}/joint_states_{idx}.txt", "w") as f:
            f.write(f"joint_states: {joint_states}\n")
            f.write(f"joint_actions: {joint_actions}\n")
        
    input("Press Enter to continue...")