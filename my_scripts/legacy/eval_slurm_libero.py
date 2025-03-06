import os
import subprocess


 
# SLURM job parameters
job_script = """#!/bin/bash
#SBATCH --job-name={model_name}_eval
#SBATCH --output=/home/daiyp/openpi/runs/logs/{model_name}_eval_{task_suite_name}_taskrange_{task_start_id}-{task_end_id}-%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1


SESSION_NAME="{model_name}_eval-{task_suite_name}-{task_suite_name}_taskrange_{task_start_id}-{task_end_id}"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
  echo "Starting new tmux session: $SESSION_NAME"
  tmux new-session -d -s $SESSION_NAME
fi

sleep 2
tmux new-window -n taskrange_{task_start_id}_{task_end_id}
tmux send-keys "/nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python /home/daiyp/openpi/scripts/serve_policy.py --env {env_name} --port {port_num}" Enter


sleep 2

/nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python  /home/daiyp/openpi/examples/libero/run_libero_eval_batch.py --task_suite_name {task_suite_name}  --task_start_id {task_start_id} --task_end_id {task_end_id} --port {port_num} --model_name {model_name}
"""

interval = 1 # run number of tasks in the batch python
total_tasks = 10
model_name = "pi0_libero" # pi0_fast_libero or pi0_libero

task_suite_name = "libero_10" # libero_object, libero_goal, libero_spatial, libero_10, libero_90



PORT_DICT = {
    "libero_object": 8000,
    "libero_goal": 8020,
    "libero_spatial": 8040,
    "libero_10": 8060,
    "libero_90": 8080,
}

ENV_NAME = {
  "pi0_fast_libero": "LIBERO_FAST",
  "pi0_libero": "LIBERO"
}
for task_start_id in range(8, 9):
# for task_start_id in range(0, total_tasks, interval):
    port_num = PORT_DICT[task_suite_name] + task_start_id
    env_name = ENV_NAME[model_name]
    task_end_id = min(task_start_id + interval, total_tasks)
    print(f"Submitting tasks {task_start_id} to {task_end_id}, using port {port_num}, task suite {task_suite_name}, model {model_name}, env {env_name}")
    
    script = job_script.format(task_start_id=task_start_id, task_end_id=task_end_id, task_suite_name=task_suite_name, port_num=port_num, model_name=model_name, env_name=env_name)
    subprocess.run(["sbatch"], input=script, text=True)
    # print(script)