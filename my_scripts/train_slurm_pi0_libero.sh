#!/bin/bash
#SBATCH --job-name=pi0_libero_ft-large_crosshair_dynamic_default_color
#SBATCH --output=/home/daiyp/openpi/runs/logs/pi0_libero_ft-large_crosshair_dynamic_default_color-%j.out
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

# # # run on their data
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_libero_finetune_bs32_acc2 --batch-size=32 --overwrite --fsdp_devices=4 --log-interval 50 --save-interval 5000 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --grad_accum_steps 2


# run on my data
XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=large_crosshair_dynamic_default_color-bs32 --batch-size=32 --overwrite --fsdp_devices=4 --log-interval 50 --save-interval 5000 --num_train_steps 32000 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --grad_accum_steps 2 --lerobot_repo_id large_crosshair_dynamic_default_color