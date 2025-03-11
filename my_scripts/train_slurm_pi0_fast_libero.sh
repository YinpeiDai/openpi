#!/bin/bash
#SBATCH --job-name=pi0_fast_libero_ft-large_crosshair_dynamic_default_color_new-resume
#SBATCH --output=/home/daiyp/openpi/runs/logs/pi0_fast_libero_ft-large_crosshair_dynamic_default_color_new-resume-%j.out
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

# # run on their data
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_fast_libero --exp-name=pi0_fast_libero_finetune_bs32 --batch-size=32 --overwrite --fsdp_devices=4 --log-interval 20 --save-interval 5000 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets


# # run on my data
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  /home/daiyp/openpi/scripts/train.py pi0_fast_libero --exp-name=pi0_fast_large_crosshair_dynamic_default_color_new --batch-size=32 --overwrite --fsdp_devices=4  --log-interval 100 --save-interval 5000  --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id large_crosshair_dynamic_default_color_new

# run on my data, resume
XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  /home/daiyp/openpi/scripts/train.py pi0_fast_libero --exp-name=pi0_fast_large_crosshair_dynamic_default_color_new --batch-size=32 --resume --fsdp_devices=4  --log-interval 100 --save-interval 5000  --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id large_crosshair_dynamic_default_color_new


# run on my data,  no delta
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python /home/daiyp/openpi/scripts/train.py pi0_fast_libero --exp-name=pi0_fast_large_crosshair_dynamic_default_color_new_no_delta --batch-size=32 --overwrite --fsdp_devices=4 --log-interval 100 --save-interval 5000 --checkpoint_base_dir /home/daiyp/openpi/runs/ckpts --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id large_crosshair_dynamic_default_color_new --no-apply-delta


# # run on my data,  no delta, resume
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  /home/daiyp/openpi/scripts/train.py pi0_fast_libero --exp-name=pi0_fast_large_crosshair_dynamic_default_color_new_no_delta --batch-size=32 --resume --fsdp_devices=4 --log-interval 100 --save-interval 5000  --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id large_crosshair_dynamic_default_color_new --no-apply-delta