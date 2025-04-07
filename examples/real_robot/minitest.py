import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import PIL

np.set_printoptions(precision=5, suppress=True)



left_shoulder_image = np.array(PIL.Image.open("/home/daiyp/openpi/examples/real_robot/reticle_samples/left_shoulder_image_111.png"))
right_shoulder_image = np.array(PIL.Image.open("/home/daiyp/openpi/examples/real_robot/reticle_samples/right_shoulder_image_111.png"))
wrist_image = np.array(PIL.Image.open("/home/daiyp/openpi/examples/real_robot/reticle_samples/wrist_image_111.png"))
instruction = "put the tennis ball in the red bowl"

state = [ 0.04678168, 0.3744241, -0.01841199, -1.66965973, -0.15927081, 2.10611606, 0.67752624, 0.63006682]
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
print(pred_action_chunk)
print(pred_action_chunk.shape)