import json
import os

import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dirname", '-d', type=str, default="all")
args = parser.parse_args()
dirname = args.dirname



def compute(dirname):
    success_rates = {}
    for filename in sorted(os.listdir(dirname)):
        if ".json" in filename:
            filepath = os.path.join(dirname, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                if "data" in data:
                    res = data["data"]
                else:
                    res = data
                if "rlbench" in filepath:
                    assert len(res) == 25
                else:
                    assert len(res) == 50
                success_rate = np.mean([1 if r["success"] else 0 for r in res])
                success_rates[filename] = success_rate
    if "rlbench" not in dirname and len(success_rates)!=10:
        print("Warning: not enough data")
    else:
        print(dirname, ':', np.mean(list(success_rates.values())))
        mean = np.mean(list(success_rates.values()))
        with open(os.path.join(dirname, f"success_rate_{mean}.txt"), "w") as f:
            for k, v in success_rates.items():
                f.write(f"{k} {v}\n")
            f.write(f"mean {np.mean(list(success_rates.values()))}\n")

if dirname == "libero":
    DIRNAME = "/home/daiyp/openpi/runs/evaluation"
    
    for d in ["libero_10", "libero_goal", "libero_object", "libero_spatial"]:
        for dd in os.listdir(os.path.join(DIRNAME, d)):
            compute(os.path.join(DIRNAME, d, dd))
elif dirname == "rlbench":
    DIRNAME = "/home/daiyp/openpi/runs/evaluation/rlbench"
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
    
    for d in RLBENCH_TASKS:
        for dd in os.listdir(os.path.join(DIRNAME, d)):
            compute(os.path.join(DIRNAME, d, dd))
    
else:
    compute(dirname)