#!/bin/bash
#SBATCH --job-name=train-pi0-libero-final_v2_large_crosshair_dynamic_default_color_long
#SBATCH --output=/home/daiyp/openpi/runs/logs/train-pi0-libero-final_v2_large_crosshair_dynamic_default_color_long-%j.out
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

source /home/daiyp/.bashrc
cd /home/daiyp/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  /home/daiyp/openpi/scripts/train.py pi0_libero --exp-name=pi0-libero-final_v2_large_crosshair_dynamic_default_color_long --batch-size=16 --overwrite --fsdp_devices=4  --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id final_v2_large_crosshair_dynamic_default_color_long --no-apply-delta --keep-period 10000


# final_large_crosshair_dynamic_plain_color, final_large_crosshair_fixed_default_color, final_small_crosshair_dynamic_default_color, final_large_bullseye_dynamic_default_color

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=final_pi0_libero_large_crosshair_dynamic_default_color --batch-size=2 --fsdp_devices=2 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id final_large_crosshair_dynamic_default_color --no-apply-delta





# XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=final_pi0_libero_large_crosshair_dynamic_default_color_no_grasp_sense --batch-size=1 --fsdp_devices=1 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id final_large_crosshair_dynamic_default_color_no_grasp_sense --no-apply-delta

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero  --exp-name=final_pi0_libero_final_large_crosshair_dynamic_plain_color --batch-size=1 --fsdp_devices=1 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id final_large_crosshair_dynamic_plain_color --no-apply-delta

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero  --exp-name=final_pi0_libero_final_large_crosshair_fixed_default_color --batch-size=1 --fsdp_devices=1 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id final_large_crosshair_fixed_default_color --no-apply-delta

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero  --exp-name=final_pi0_libero_final_small_crosshair_dynamic_default_color --batch-size=1 --fsdp_devices=1 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id final_small_crosshair_dynamic_default_color --no-apply-delta

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero  --exp-name=final_pi0_libero_final_large_bullseye_dynamic_default_color --batch-size=1 --fsdp_devices=1 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id final_large_bullseye_dynamic_default_color --no-apply-delta