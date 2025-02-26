#!/bin/bash
#SBATCH --job-name=pi0_fast_libero_finetune
#SBATCH --output=/home/daiyp/openpi/logs/pi0_fast_libero_finetune-%j.out
#SBATCH --gres=gpu:4
#SBATCH --time=12-00:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --mem-per-gpu=46G
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

source /home/daiyp/.bashrc
cd /home/daiyp/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_fast_libero --exp-name=pi0_fast_libero_finetune_bs32 --batch-size=32 --overwrite --fsdp_devices=4 --log-interval 20 --save-interval 5000 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets
