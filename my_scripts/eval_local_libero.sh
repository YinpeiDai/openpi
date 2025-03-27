# server
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --env LIBERO_FAST  --port 8000

XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8012  policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_fast_libero/pi0_fast_libero_finetune_bs32/30000


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8012  policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_fast_libero/pi0_fast_libero_finetune_bs32/30000


# greatlakes
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/serve_policy.py --port 8012  --lerobot_repo_id  libero_large_crosshair_dynamic_default_color_grpsen  --no-apply-delta   policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=/home/daiyp/openpi/runs/ckpts/pi0_libero/pi0_libero_large_crosshair_dynamic_default_color_grpsen/25000




# client
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/scopereticle/src
# not use reticle
CUDA_VISIBLE_DEVICES=0 python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 10 --port 8012 
# use reticle
CUDA_VISIBLE_DEVICES=0 python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 10 --port 8012 --use_reticle --reticle_config_key large_crosshair_dynamic_default_color


CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python examples/libero/run_libero_eval_batch.py  --model-name testestest --task_suite_name libero_spatial  --task_start_id 0 --task_end_id 1 --port 8012 --use_reticle --reticle_config_key large_crosshair_dynamic_default_color --use_grasp_sense


