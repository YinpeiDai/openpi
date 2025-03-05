# server
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --env LIBERO_FAST  --port 8000

XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8012  policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_fast_libero/pi0_fast_libero_finetune_bs32/30000





# client
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/scopereticle/src
# not use reticle
CUDA_VISIBLE_DEVICES=0 python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 10 --port 8012 
# use reticle
CUDA_VISIBLE_DEVICES=0 python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 10 --port 8012 --use_reticle --reticle_config_key large_crosshair_dynamic_default_color
