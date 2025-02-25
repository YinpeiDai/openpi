# dirname = "/home/daiyp/openvla-mini/runs/libero_90/minivla-wrist-vq-libero90-prismatic"
dirname = "/home/daiyp/openvla-mini/runs/libero_spatial/openvla-7b-finetuned-libero-spatial"
dirname = "/home/daiyp/openvla-mini/gl_results/libero_90/minivla-vq-libero90-prismatic"

dirname = "/home/daiyp/openpi/data/libero_10/pi0_libero"
# dirname = "/home/daiyp/openpi/data/libero_object/pi0_libero"
# dirname = "/home/daiyp/openpi/data/libero_goal/pi0_libero"
# dirname = "/home/daiyp/openpi/data/libero_spatial/pi0_libero"


import json
import os

import numpy as np

success_rates = []
for filename in sorted(os.listdir(dirname)):
    if ".json" in filename:
        with open(os.path.join(dirname, filename), "r") as f:
            data = json.load(f)
            res = data["data"]
            assert len(res) == 50
            success_rate = np.mean([1 if r["success"] else 0 for r in res])
            print(filename, success_rate)
            success_rates.append(success_rate)

print(np.mean(success_rates))