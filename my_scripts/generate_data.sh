# replay libero data to get no_noops data
# see openvla

# transfer rlds data into lerobot data
# change REPO_NAME (for lerobot repo id), RAW_DATASET_NAMES (for  /path/to/your/libero/data)

micromamba activate tfds2lerobot
python examples/libero/convert_libero_data_to_lerobot.py --data_dir $MYDIR/tfds_reticle_data/final_large_crosshair_dynamic_default_color_no_grasp_sense --repo_id large_crosshair_dynamic_default_color_no_grasp_sense

python examples/libero/convert_libero_data_to_lerobot.py --data_dir $MYDIR/tfds_reticle_data/final_large_crosshair_dynamic_plain_color --repo_id large_crosshair_dynamic_plain_color

python examples/libero/convert_libero_data_to_lerobot.py --data_dir $MYDIR/tfds_reticle_data/final_large_crosshair_fixed_default_color --repo_id large_crosshair_fixed_default_color

python examples/libero/convert_libero_data_to_lerobot.py --data_dir $MYDIR/tfds_reticle_data/final_small_crosshair_dynamic_default_color --repo_id small_crosshair_dynamic_default_color

python examples/libero/convert_libero_data_to_lerobot.py --data_dir $MYDIR/tfds_reticle_data/final_large_bullseye_dynamic_default_color --repo_id large_bullseye_dynamic_default_color

python examples/libero/convert_libero_data_to_lerobot.py --data_dir /home/ubuntu/chailab/daiyp/tfds_reticle_data/final_large_crosshair_dynamic_default_color_tilt --repo_id xxx





large_bullseye_dynamic_default_color
large_crosshair_dynamic_default_color
large_crosshair_dynamic_default_color_no_grasp_sense
large_crosshair_dynamic_plain_color
large_crosshair_fixed_default_color
small_crosshair_dynamic_default_color
large_crosshair_dynamic_default_color_tilt


uv run scripts/compute_norm_stats.py --config-name pi0_fast_realrobot --lerobot-repo-id realrobot_cup_in_coffee_machine_reticle 

python examples/real_robot/convert_hdf5_data_to_lerobot.py  --data-dir /data/daiyp/crosshair/real_data/coffee --repo-id cup_in_coffee_machine  --use-reticle

uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero --lerobot-repo-id final_v2_large_crosshair_dynamic_default_color_long 


# comute stats for my own dataset
XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero --lerobot-repo-id large_crosshair_dynamic_default_color 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 uv run scripts/compute_norm_stats.py --config-name pi0_libero --lerobot-repo-id large_crosshair_dynamic_default_color 


XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python scripts/compute_norm_stats.py --config-name pi0_libero --lerobot-repo-id large_crosshair_dynamic_default_color_no_grasp_sense

XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /home/daiyp/openpi/tensorflow_datasets/final_v2_small_crosshair_dynamic_default_color --repo_id final_v2_small_crosshair_dynamic_default_color

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