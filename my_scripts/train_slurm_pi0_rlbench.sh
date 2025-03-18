#!/bin/bash
#SBATCH --job-name=pi0_rlbench_grpsen_joinpos
#SBATCH --output=/home/daiyp/openpi/runs/logs/pi0_rlbench_grpsen_joinpos-%j.out
#SBATCH --gres=gpu:4
#SBATCH --time=12-00:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

source /home/daiyp/.bashrc
cd /home/daiyp/openpi


XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_rlbench --exp-name=pi0_rlbench_grpsen_joinpos --batch-size=32 --overwrite --fsdp_devices=4   --grad_accum_steps 2 --lerobot_repo_id rlbench_grpsen_joinpos



# XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_rlbench --exp-name=pi0_rlbench_large_crosshair_dynamic_default_color_grpsen_joinpos --batch-size=32 --overwrite --fsdp_devices=4  --grad_accum_steps 2 --lerobot_repo_id rlbench_large_crosshair_dynamic_default_color_grpsen_joinpos