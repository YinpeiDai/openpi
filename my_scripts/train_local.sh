# replay libero data to get no_noops data
# see openvla

# transfer rlds data into lerobot data
# change REPO_NAME (for lerobot repo id), RAW_DATASET_NAMES (for  /path/to/your/libero/data)
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /nfs/turbo/coe-chaijy-unreplicated/daiyp/tensorflow_datasets/large_crosshair_dynamic_default_color --repo_id testest

# comute stats for the dataset
uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero --lerobot-repo-id large_crosshair_dynamic_default_color 


XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_libero --exp-name=my_experiment --batch-size=8 --overwrite --fsdp_devices=2

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run  scripts/train.py pi0_libero --exp-name=testest --batch-size=2 --overwrite --fsdp_devices=2 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets