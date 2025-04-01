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


