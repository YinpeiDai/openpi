# server
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --env LIBERO_FAST  --port 8000

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8001  --policy_config pi0_fast_libero --policy_dir /nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_fast_libero/pi0_fast_libero_finetune_bs32/30000

# client
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/run_libero_eval_batch.py  --model-name rerun_release_pi0fast --task_suite_name libero_goal  --task_start_id 0 --task_end_id 10 --port 8000
python examples/libero/run_libero_eval_batch.py  --model-name pi0-fast-ckpt30000 --task_suite_name libero_goal  --task_start_id 0 --task_end_id 10 --port 8001
