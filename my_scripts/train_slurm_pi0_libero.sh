#!/bin/bash
#SBATCH --job-name=pi0_modified_libero_rlds_no_delta-resume
#SBATCH --output=/home/daiyp/openpi/runs/logs/pi0_modified_libero_rlds_no_delta-resume-%j.out
#SBATCH --gres=gpu:4
#SBATCH --time=12-00:00:00
#SBATCH --account=nfz0
#SBATCH --partition=spgpu
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

source /home/daiyp/.bashrc
cd /home/daiyp/openpi

# # # run on their data
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_libero_finetune_bs32_acc2 --batch-size=32 --overwrite --fsdp_devices=4 --log-interval 50 --save-interval 5000 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --grad_accum_steps 2

# run on my data
XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_modified_libero_rlds_no_delta --batch-size=32 --resume --fsdp_devices=4 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --grad_accum_steps 2 --lerobot_repo_id modified_libero_rlds --no-apply-delta


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_libero_large_crosshair_dynamic_default_color_grpsen --batch-size=32 --overwrite --fsdp_devices=4  --checkpoint_base_dir /home/daiyp/openpi/runs/ckpts  --num_train_steps 35000   --assets_base_dir /home/daiyp/openpi/runs/assets --grad_accum_steps 2 --lerobot_repo_id libero_large_crosshair_dynamic_default_color_grpsen  --no-apply-delta
