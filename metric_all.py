import json
import os

import numpy as np

import argparse

dirname = "runs/evaluation"
task_suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
results = {"libero_spatial": {}, "libero_object": {}, "libero_goal": {}, "libero_10": {}}





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
                assert len(res) == 50
                success_rate = np.mean([1 if r["success"] else 0 for r in res])
                success_rates[filename] = success_rate
    if "rlbench" not in dirname and len(success_rates)!=10:
        print("Warning: not enough data")
        return np.nan
    else:
        print(dirname, ':', np.mean(list(success_rates.values())))
        mean = np.mean(list(success_rates.values()))
        with open(os.path.join(dirname, f"success_rate_{mean}.txt"), "w") as f:
            for k, v in success_rates.items():
                f.write(f"{k} {v}\n")
            f.write(f"mean {np.mean(list(success_rates.values()))}\n")
    
        return np.mean(list(success_rates.values()))

for task_suite in task_suites:
    for filename in sorted(os.listdir(os.path.join(dirname, task_suite))):
        if "pi0" in filename:
            filepath = os.path.join(dirname, task_suite, filename)
            results[task_suite][filename] = compute(filepath)

# print results in table format
# model, libero_10, libero_goal, libero_object, libero_spatial
#  xxx, 0.95, 0.96, 0.97, 0.98

# use pandas to print results in table format
import pandas as pd

# Set display options to show full strings
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

df = pd.DataFrame(results)
df['mean'] = df.mean(axis=1) 
# save 3 decimal places
df = df.round(3)
print(df)
