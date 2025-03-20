# replay libero data to get no_noops data
# see openvla

# transfer rlds data into lerobot data
# change REPO_NAME (for lerobot repo id), RAW_DATASET_NAMES (for  /path/to/your/libero/data)
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /nfs/turbo/coe-chaijy-unreplicated/daiyp/tensorflow_datasets/large_crosshair_dynamic_default_color --repo_id testest

# comute stats for my own dataset
uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero --lerobot-repo-id large_crosshair_dynamic_default_color 
uv run scripts/compute_norm_stats.py --config-name pi0_libero --lerobot-repo-id large_crosshair_dynamic_default_color 



# rlbench
uv run scripts/compute_norm_stats.py --config-name pi0_fast_rlbench --lerobot-repo-id rlbench_grpsen_joinpos

XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/compute_norm_stats.py --config-name pi0_libero --lerobot-repo-id libero_large_crosshair_dynamic_default_color_grpsen

XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/compute_norm_stats.py --config-name pi0_fast_libero --lerobot-repo-id libero_large_crosshair_dynamic_default_color_grpsen


XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/compute_norm_stats.py --config-name pi0_rlbench --lerobot-repo-id rlbench_large_crosshair_dynamic_default_color_grpsen_joinpos 

XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=1 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/compute_norm_stats.py --config-name pi0_rlbench --lerobot-repo-id rlbench_grpsen_joinpos 

XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=2 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/compute_norm_stats.py --config-name pi0_fast_rlbench --lerobot-repo-id rlbench_grpsen_joinpos 

XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=3 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/compute_norm_stats.py --config-name pi0_fast_rlbench --lerobot-repo-id rlbench_large_crosshair_dynamic_default_color_grpsen_joinpos 


XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/compute_norm_stats.py --config-name pi0_fast_libero --lerobot-repo-id libero_large_crosshair_dynamic_default_color_grpsen 


XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/compute_norm_stats.py --config-name pi0_libero --lerobot-repo-id libero_large_crosshair_dynamic_default_color_grpsen --no-apply-delta