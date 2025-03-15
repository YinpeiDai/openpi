import os
import subprocess
import socket


 
# SLURM job parameters
job_script = """#!/bin/bash
#SBATCH --job-name=eval_{model_name}
#SBATCH --output=/home/daiyp/openpi/runs/logs/eval_{model_name}_{task_name}-%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --mem-per-gpu=46G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

cd /home/daiyp/openpi

SESSION_NAME="Eval-{model_name}-{task_suite_name}"

for port in $(seq 8000 9000); do
    if ! ss -tuln | grep -q ":$port "; then
        echo "Available port: $port"
        break
    fi
done


tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
  echo "Starting new tmux session: $SESSION_NAME"
  tmux new-session -d -s $SESSION_NAME
fi

sleep 2
tmux new-window -n server
tmux send-keys "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python /home/daiyp/openpi/scripts/serve_policy.py --port $port --lerobot_repo_id {lerobot_repo_id}  policy:checkpoint --policy.config={policy_config} --policy.dir={ckpt_dir}" Enter
sleep 2

"""

eval_script_reticle = """
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/scopereticle/src
/nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python  examples/libero/run_libero_eval_batch.py  --model-name {model_name} --task_suite_name {task_suite_name}  --port $port --use_reticle --reticle_config_key {reticle_cfg} --task_start_id {task_start_id} --task_end_id {task_end_id}

"""

eval_script_no_reticle = """

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/scopereticle/src
/nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python  examples/libero/run_libero_eval_batch.py  --model-name {model_name} --task_suite_name {task_suite_name}  --port $port --task_start_id {task_start_id} --task_end_id {task_end_id}
"""

# Evaluate on 10 tasks
model_name = "pi0_libero_large_crosshair_dynamic_default_color_ckpt15000"
policy_config = "pi0_libero"
ckpt_dir="/home/daiyp/openpi/runs/ckpts/pi0_libero/pi0_large_crosshair_dynamic_default_color/15000"
use_reticle=1
reticle_cfg="large_crosshair_dynamic_default_color" 
lerobot_repo_id="large_crosshair_dynamic_default_color"  # "physical-intelligence/libero"
task_start_id = 0
task_end_id = 10


# eval on one GPU for task_suite_name and task id
for task_suite_name in ["libero_10", " libero_object", "libero_goal", "libero_spatial"]:
    if use_reticle:
        print(f"Eval {policy_config}: \n  Submitting tasks using task suite [{task_suite_name}]\n  model {model_name} \n  with reticle_config '{reticle_cfg}'")
        script = job_script.format(model_name=model_name, task_suite_name=task_suite_name, ckpt_dir=ckpt_dir, policy_config=policy_config, task_start_id=task_start_id, task_end_id=task_end_id, lerobot_repo_id=lerobot_repo_id) + \
            eval_script_reticle.format(model_name=model_name, task_suite_name=task_suite_name, reticle_cfg=reticle_cfg, task_start_id=task_start_id, task_end_id=task_end_id)
    else:
        print(f"Eval {policy_config}: \n  Submitting tasks using task suite [{task_suite_name}]\n  model {model_name} \n  without reticle")
        script = job_script.format(model_name=model_name, task_suite_name=task_suite_name, ckpt_dir=ckpt_dir, policy_config=policy_config, task_start_id=task_start_id, task_end_id=task_end_id)+ \
            eval_script_no_reticle.format(model_name=model_name, task_suite_name=task_suite_name, task_start_id=task_start_id, task_end_id=task_end_id)
    subprocess.run(["sbatch"], input=script, text=True)
    print('---')


# # eval each GPU on one task
# for task_suite_name in ["libero_10", " libero_object", "libero_goal", "libero_spatial"]:
#     for task_start_id in range(10):
#         task_end_id = task_start_id + 1
#         if use_reticle:
#             print(f"Eval {policy_config}: \n  Submitting tasks using task suite [{task_suite_name}]\n  model {model_name} \n  with reticle_config '{reticle_cfg}'")
#             script = job_script.format(model_name=model_name, task_suite_name=task_suite_name, ckpt_dir=ckpt_dir, policy_config=policy_config, task_start_id=task_start_id, task_end_id=task_end_id, lerobot_repo_id=lerobot_repo_id) + \
#                 eval_script_reticle.format(model_name=model_name, task_suite_name=task_suite_name, reticle_cfg=reticle_cfg, task_start_id=task_start_id, task_end_id=task_end_id)
#         else:
#             print(f"Eval {policy_config}: \n  Submitting tasks using task suite [{task_suite_name}]\n  model {model_name} \n  without reticle")
#             script = job_script.format(model_name=model_name, task_suite_name=task_suite_name, ckpt_dir=ckpt_dir, policy_config=policy_config, task_start_id=task_start_id, task_end_id=task_end_id)+ \
#                 eval_script_no_reticle.format(model_name=model_name, task_suite_name=task_suite_name, task_start_id=task_start_id, task_end_id=task_end_id)
#         subprocess.run(["sbatch"], input=script, text=True)
#         print('---')





