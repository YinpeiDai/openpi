# server
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 $MYDIR/micromamba/envs/openpi/bin/python scripts/serve_policy.py --env LIBERO_FAST  --port 8000  policy:checkpoint --policy.config=pi0_fast_libero  --policy.dir=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_fast_libero


# client
source setup_libero.sh

# not use reticle
python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 10 --port 8000 --save_path $MYDIR/openpi/runs/evaluation

# use reticle
CUDA_VISIBLE_DEVICES=0 python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 10 --port 8012 --use_reticle --reticle_config_key large_crosshair_dynamic_default_color


CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 1 --port 8012 --use_reticle --reticle_config_key large_crosshair_dynamic_default_color --use_grasp_sense


