#!/bin/bash
#SBATCH --job-name=pi0_fast-realrobot_pickpalce_10demos_reticle
#SBATCH --output=/home/daiyp/openpi/runs/logs/pi0_fast-realrobot_pickpalce_10demos_reticle-%j.out
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

source /home/daiyp/.bashrc
cd /home/daiyp/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python /home/daiyp/openpi/scripts/train.py pi0_fast_realrobot  --exp-name=pi0_fast-realrobot_pickpalce_10demos_reticle --batch-size=16 --overwrite --fsdp_devices=4 --lerobot_repo_id realrobot_pickpalce_10demos_reticle

