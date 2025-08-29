import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import PIL

np.set_printoptions(precision=5, suppress=True)



left_shoulder_image = np.array(PIL.Image.open("/home/daiyp/openpi/examples/real_robot/reticle_samples_fruits/left_shoulder_image_120.png"))
right_shoulder_image = np.array(PIL.Image.open("/home/daiyp/openpi/examples/real_robot/reticle_samples_fruits/right_shoulder_image_120.png"))
wrist_image = np.array(PIL.Image.open("/home/daiyp/openpi/examples/real_robot/reticle_samples_fruits/wrist_image_120.png"))
instruction = "put all the fruits from the basket into the box"

state = [ 0.04437614, -0.22375874,  0.08395074, -2.55443907,  0.15480031,  2.19466829,  0.41659123,  0.99954033]
state = np.array(state, dtype=np.float32)

request_data = {
    "observation/left_shoulder_image": image_tools.resize_with_pad(left_shoulder_image, 224, 224),  
    "observation/right_shoulder_image": image_tools.resize_with_pad(right_shoulder_image, 224, 224),
    "observation/wrist_image": image_tools.resize_with_pad(wrist_image, 224, 224),
    "observation/state": state,
    "prompt": instruction,
}

policy_client = websocket_client_policy.WebsocketClientPolicy(
    host="141.212.115.116",
    port=8001,
)
tstart = time.time()
pred_action_chunk = policy_client.infer(request_data)["actions"]
tend = time.time()
print(f"Time taken: {tend - tstart} seconds")
print(pred_action_chunk[0]) # [ 0.04234172 -0.23474922  0.11579343 -2.53332103  0.16039336  2.16150408 0.52410762  1.        ]
print(pred_action_chunk.shape)