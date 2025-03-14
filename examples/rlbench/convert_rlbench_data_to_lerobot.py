
import os
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro
from PIL import Image

np.set_printoptions(precision=4, suppress=True)

# close_jar_demo.hdf5 100
# insert_onto_square_peg_demo.hdf5 100
# light_bulb_in_demo.hdf5 98
# meat_off_grill_demo.hdf5 100
# open_drawer_demo.hdf5 98
# push_buttons_demo.hdf5 100
# stack_blocks_demo.hdf5 85
# stack_cups_demo.hdf5 98

def main(data_dir: str = "/home/daiyp/openpi/rlbench_regenerate_data", repo_id: str = "rlbench", *, push_to_hub: bool = False):
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
        fps=10,
        features={
            "image": {
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
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in sorted(os.listdir(data_dir)):
        if ".hdf5" not in raw_dataset_name:
            continue
        data_path = os.path.join(data_dir, raw_dataset_name)
        data_file = h5py.File(data_path, "r")
        orig_data = data_file["data"]
        # print()
        
        for ep in orig_data:
            demo_data = orig_data[ep]
            # print("demo_data:", demo_data.keys())
            orig_actions = demo_data["actions"][()]
            orig_imges = demo_data["obs"]["front_images"][()]
            orig_wrist_images = demo_data["obs"]["wrist_images"][()]
            
            orig_imges_reticle = demo_data["obs"]["front_images_reticle"][()]
            orig_wrist_images_reticle = demo_data["obs"]["wrist_images_reticle"][()]
            
            ee_states = demo_data["obs"]["ee_states"][()]
            gripper_states = demo_data["obs"]["gripper_states"][()]
            joint_states = demo_data["obs"]["joint_states"][()]
            
            language_instruction = demo_data["language_instruction"][()].decode()
            
            # print("language_instruction:", language_instruction)
            # print("orig_actions:", orig_actions.shape)
            # print("orig_imges:", orig_imges.shape)
            # print("orig_wrist_images:", orig_wrist_images.shape)
            # print("ee_states:", ee_states.shape)
            # print("gripper_states:", gripper_states.shape)
            
            
            for step in range(len(orig_actions)):
                # Image.fromarray(orig_imges_reticle[step]).save(f"img.png")
                # Image.fromarray(orig_wrist_images_reticle[step]).save(f"wrist_img.png")
                # print(np.asarray(np.concatenate([ee_states[step], gripper_states[step]], axis=-1), dtype=np.float32))
                # print(np.asarray(orig_actions[step], dtype=np.float32))
                dataset.add_frame(
                    {
                        "image": orig_imges[step],
                        "wrist_image": orig_wrist_images[step],
                        "state": np.asarray(np.concatenate([ee_states[step], gripper_states[step]], axis=-1), dtype=np.float32),
                        "actions": np.asarray(orig_actions[step], dtype=np.float32),
                    }
                )
                # input("continue?")
            dataset.save_episode(task=language_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
