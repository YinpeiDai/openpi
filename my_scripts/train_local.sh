XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_libero --exp-name=my_experiment --batch-size=8 --overwrite --fsdp_devices=2

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run  scripts/train.py pi0_libero --exp-name=testest --batch-size=2 --overwrite --fsdp_devices=2 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets