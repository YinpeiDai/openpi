large_bullseye_dynamic_default_color
large_crosshair_dynamic_default_color
large_crosshair_dynamic_default_color_no_grasp_sense
large_crosshair_dynamic_plain_color
large_crosshair_fixed_default_color
small_crosshair_dynamic_default_color


# pi0_fast_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 $MYDIR/micromamba/envs/openpi/bin/python scripts/train.py pi0_fast_libero --exp-name=lambda-pi0-fast-libero-large_crosshair_dynamic_default_color --batch-size=32 --overwrite --lerobot_repo_id large_crosshair_dynamic_default_color --no-apply-delta --num_workers=0


# pi0_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 $MYDIR/micromamba/envs/openpi/bin/python scripts/train.py pi0_libero --exp-name=lambda-pi0-libero-large_crosshair_dynamic_plain_color --batch-size=32 --overwrite --lerobot_repo_id large_crosshair_dynamic_plain_color --no-apply-delta --num_workers=0



CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run  scripts/train.py pi0_libero --exp-name=testest --batch-size=2 --overwrite --fsdp_devices=2 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets



##### run on my data #####

# pi0_fast_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_libero --exp-name=large_crosshair_dynamic_default_color --batch-size=2 --overwrite --fsdp_devices=2  --num_train_steps 32000 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id large_crosshair_dynamic_default_color

# pi0_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_libero --exp-name=large_crosshair_dynamic_default_color --batch-size=2 --overwrite --fsdp_devices=2  --num_train_steps 32000 --checkpoint_base_dir  /home/daiyp/openpi/runs/ckpts   --assets_base_dir /home/daiyp/openpi/runs/assets --lerobot_repo_id large_crosshair_dynamic_default_color


# run on real robot
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run scripts/train.py pi0_fast_realrobot --exp-name=xxxxx --batch-size=4 --overwrite --fsdp_devices=2 --lerobot_repo_id realrobot_pickpalce_10demos