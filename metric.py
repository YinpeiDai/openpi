import json
import os

import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dirname", '-d', type=str, required=True)
args = parser.parse_args()
dirname = args.dirname


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