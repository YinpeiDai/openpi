#!/bin/bash
#SBATCH --job-name=pi0_libero_8gpu
#SBATCH --output=/home/daiyp/openpi/runs/logs/pi0_libero_8gpu-%j.out
#SBATCH --gres=gpu:8
#SBATCH --time=14-00:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --mem=320G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1


source /home/daiyp/.bashrc
cd /home/daiyp/openpi


XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_libero_8gpu_final_large_crosshair_dynamic_default_color --batch-size=32 --overwrite --fsdp_devices=8  --lerobot_repo_id final_large_crosshair_dynamic_default_color  --no-apply-delta

sleep 10

XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_libero_8gpu_final_large_crosshair_dynamic_default_color_no_grasp_sense --batch-size=32 --overwrite --fsdp_devices=8  --lerobot_repo_id final_large_crosshair_dynamic_default_color_no_grasp_sense  --no-apply-delta

sleep 10

XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_libero_8gpu_final_small_crosshair_dynamic_default_color --batch-size=32 --overwrite --fsdp_devices=8  --lerobot_repo_id final_small_crosshair_dynamic_default_color  --no-apply-delta

sleep 10

XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_libero_8gpu_final_large_crosshair_dynamic_plain_color --batch-size=32 --overwrite --fsdp_devices=8  --lerobot_repo_id final_large_crosshair_dynamic_plain_color  --no-apply-delta

sleep 10


XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_libero_8gpu_final_large_crosshair_fixed_default_color --batch-size=32 --overwrite --fsdp_devices=8   --lerobot_repo_id final_large_crosshair_fixed_default_color  --no-apply-delta

sleep 10

XLA_PYTHON_CLIENT_MEM_FRACTION=0.975 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python  scripts/train.py pi0_libero --exp-name=pi0_libero_8gpu_final_large_bullseye_dynamic_default_color --batch-size=32 --overwrite --fsdp_devices=8  --lerobot_repo_id final_large_bullseye_dynamic_default_color  --no-apply-delta




