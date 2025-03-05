# replay libero data to get no_noops data
# see openvla

# transfer rlds data into lerobot data
# change REPO_NAME (for lerobot repo id), RAW_DATASET_NAMES (for  /path/to/your/libero/data)
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /nfs/turbo/coe-chaijy-unreplicated/daiyp/tensorflow_datasets/large_crosshair_dynamic_default_color --repo_id testest

# comute stats for my own dataset
uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero --lerobot-repo-id large_crosshair_dynamic_default_color 
uv run scripts/compute_norm_stats.py --config-name pi0_libero --lerobot-repo-id large_crosshair_dynamic_default_color 

