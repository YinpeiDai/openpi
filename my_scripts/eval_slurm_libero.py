import os
import subprocess
import socket


# SLURM job parameters
job_script = """#!/bin/bash
#SBATCH --job-name=eval_{model_name}
#SBATCH --output=/home/daiyp/openpi/runs/logs/eval_{model_name}_{task_suite_name}-task{task_start_id}_{task_end_id}-port{port}-%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --mem-per-gpu=46G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

cd /home/daiyp/openpi

module load tmux/3.3a

SESSION_NAME="Eval-{model_name}-{task_suite_name}-task{task_start_id}_{task_end_id}_{port}"

echo "Starting new tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

set -x

sleep 2
tmux new-window -n server
tmux send-keys "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python /home/daiyp/openpi/scripts/serve_policy.py --port {port} --lerobot_repo_id {lerobot_repo_id} {apply_delta_cmd} policy:checkpoint --policy.config={policy_config} --policy.dir={ckpt_dir}" Enter
sleep 2

"""

eval_script_reticle = """

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/scopereticle/src
/nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python  examples/libero/run_libero_eval_batch.py  --model-name {model_name} --task_suite_name {task_suite_name}  --port {port} --use_reticle --reticle_config_key {reticle_cfg} --task_start_id {task_start_id} --task_end_id {task_end_id}"""

eval_script_no_reticle = """

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/scopereticle/src
/nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python  examples/libero/run_libero_eval_batch.py  --model-name {model_name} --task_suite_name {task_suite_name}  --port {port} --task_start_id {task_start_id} --task_end_id {task_end_id}"""

# Evaluate on 10 tasks
model_type = "pi0_fast_libero"
model_name = "pi0-fast-libero-large_crosshair_dynamic_default_color_new-rerun-ckpt20k"
policy_config = model_type
ckpt_dir="/home/daiyp/openpi/runs/ckpts/pi0_fast_libero/pi0-fast-libero-large_crosshair_dynamic_default_color_new-rerun/20000"
port_base=8300 # !!!!!!
use_reticle=1
reticle_cfg="large_crosshair_dynamic_default_color"
lerobot_repo_id="large_crosshair_dynamic_default_color_new"
apply_delta=False #!!!!!!

use_grasp_sense=False
if use_grasp_sense:
    use_grasp_sense_cmd = " --use_grasp_sense"
else:
    use_grasp_sense_cmd = ""

if not apply_delta:
    apply_delta_cmd = " --no-apply-delta"
else:
    apply_delta_cmd = ""

# eval each GPU on one task
port = port_base
for task_suite_name in [
    "libero_spatial", 
    "libero_object", 
    "libero_goal", 
    "libero_10"
    ]:
    for task_start_id in range(0, 1):
        port += 1
        task_end_id = 10 #task_start_id + 1
        if use_reticle:
            print(f"Eval {policy_config}: \n  Submitting tasks using task suite [{task_suite_name}], port={port}\n  model {model_name} \n  with reticle_config '{reticle_cfg}' \n  apply_delta={apply_delta}")
            script = job_script.format(model_name=model_name, task_suite_name=task_suite_name, ckpt_dir=ckpt_dir, policy_config=policy_config, task_start_id=task_start_id, task_end_id=task_end_id, lerobot_repo_id=lerobot_repo_id, port=port, apply_delta_cmd=apply_delta_cmd) + \
                eval_script_reticle.format(model_name=model_name, task_suite_name=task_suite_name, reticle_cfg=reticle_cfg, task_start_id=task_start_id, task_end_id=task_end_id, port=port)
            if use_grasp_sense:
                script = script + use_grasp_sense_cmd
        else:
            print(f"Eval {policy_config}: \n  Submitting tasks using task suite [{task_suite_name}], port={port}\n  model {model_name} \n  without reticle \n  apply_delta={apply_delta}")
            script = job_script.format(model_name=model_name, task_suite_name=task_suite_name, ckpt_dir=ckpt_dir, policy_config=policy_config, task_start_id=task_start_id, task_end_id=task_end_id, lerobot_repo_id=lerobot_repo_id, port=port, apply_delta_cmd=apply_delta_cmd)+ \
                eval_script_no_reticle.format(model_name=model_name, task_suite_name=task_suite_name, task_start_id=task_start_id, task_end_id=task_end_id, port=port)
            if use_grasp_sense:
                script = script + use_grasp_sense_cmd
        subprocess.run(["sbatch"], input=script, text=True)
        
        # print(script)
        # input('---')






# # eval on one GPU for task_suite_name and task id
# task_start_id = 0
# task_end_id = 10
# for idx, task_suite_name in enumerate([
#     "libero_10", 
#     # "libero_object", 
#     # "libero_goal", 
#     # "libero_spatial"
# ]):
#     port = port_base + idx
#     if use_reticle:
#         print(f"Eval {policy_config}: \n  Submitting tasks using task suite [{task_suite_name}], port={port}\n  model {model_name} \n  with reticle_config '{reticle_cfg}'")
#         script = job_script.format(model_name=model_name, task_suite_name=task_suite_name, ckpt_dir=ckpt_dir, policy_config=policy_config, task_start_id=task_start_id, task_end_id=task_end_id, lerobot_repo_id=lerobot_repo_id, port=port) + \
#             eval_script_reticle.format(model_name=model_name, task_suite_name=task_suite_name, reticle_cfg=reticle_cfg, task_start_id=task_start_id, task_end_id=task_end_id, port=port)
#     else:
#         print(f"Eval {policy_config}: \n  Submitting tasks using task suite [{task_suite_name}], port={port}\n  model {model_name} \n  without reticle")
#         script = job_script.format(model_name=model_name, task_suite_name=task_suite_name, ckpt_dir=ckpt_dir, policy_config=policy_config, task_start_id=task_start_id, task_end_id=task_end_id, lerobot_repo_id=lerobot_repo_id, port=port)+ \
#             eval_script_no_reticle.format(model_name=model_name, task_suite_name=task_suite_name, task_start_id=task_start_id, task_end_id=task_end_id, port=port)
#     subprocess.run(["sbatch"], input=script, text=True)
#     print('---')
