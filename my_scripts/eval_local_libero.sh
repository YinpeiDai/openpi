# server
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 $MYDIR/micromamba/envs/openpi/bin/python scripts/serve_policy.py --env LIBERO_FAST  --port 8000  policy:checkpoint --policy.config=pi0_libero  --policy.dir=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_libero


XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 $MYDIR/micromamba/envs/openpi/bin/python scripts/serve_policy.py --port 8000  --lerobot-repo-id large_crosshair_dynamic_default_color  --no-apply-delta   policy:checkpoint --policy.config=pi0_libero  --policy.dir=$MYDIR/openpi/runs/ckpts/pi0_libero/lambda-pi0-libero-large_crosshair_dynamic_default_color/29999

XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 /home/ubuntu/chailab/daiyp/micromamba/envs/openpi/bin/python scripts/serve_policy.py --port 8000  --lerobot-repo-id final_v2_large_crosshair_dynamic_default_color  --no-apply-delta   policy:checkpoint --policy.config=pi0_libero  --policy.dir=/home/ubuntu/chailab/daiyp/openpi/runs/ckpts/pi0_libero/lambda-pi0-libero-final_v2_large_crosshair_dynamic_default_color/20000


# client
source setup_libero.sh

# not use reticle
python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 10 --port 8000 --use_reticle --reticle_config_key large_crosshair_dynamic_default_color --use_grasp_sense

# use reticle
CUDA_VISIBLE_DEVICES=0 python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 10 --port 8012 --use_reticle --reticle_config_key large_crosshair_dynamic_default_color


CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 1 --port 8012 --use_reticle --reticle_config_key large_crosshair_dynamic_default_color --use_grasp_sense


