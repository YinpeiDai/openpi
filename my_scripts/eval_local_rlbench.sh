# server
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --env LIBERO_FAST  --port 8000

XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8012  --no-apply-delta --lerobot_repo_id rlbench  policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=/home/daiyp/openpi/runs/ckpts/pi0_fast_libero/pi0_fast_rlbench/20000 


XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port 8013  --no-apply-delta --lerobot_repo_id rlbench_large_crosshair_dynamic_default_color  policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=/home/daiyp/openpi/runs/ckpts/pi0_fast_libero/pi0_fast_rlbench_large_crosshair_dynamic_default_color/18000


XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/serve_policy.py --port 8012  --no-apply-delta --lerobot_repo_id rlbench  policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=/home/daiyp/openpi/runs/ckpts/pi0_fast_libero/pi0_fast_rlbench/20000 


# greatlakes
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi/bin/python scripts/serve_policy.py --port 8000  --lerobot_repo_id rlbench_grpsen_joinpos  policy:checkpoint --policy.config=pi0_fast_rlbench --policy.dir=/home/daiyp/openpi/runs/ckpts/pi0_fast_rlbench/pi0_fast_rlbench_grpsen_joinpos/15000 


# client
source examples/rlbench/.venv/bin/activate

source examples/rlbench/setup_bash.sh
# not use reticle
CUDA_VISIBLE_DEVICES=0 python examples/rlbench/run_rlbench_eval_batch.py  --model-name testestest --task_name close_jar  --port 8012 
# use reticle
CUDA_VISIBLE_DEVICES=1 python examples/rlbench/run_rlbench_eval_batch.py  --model-name comeoncomeon --task_name close_jar  --port 8013  --use_reticle --reticle_config_key large_crosshair_dynamic_default_color


# CUDA_VISIBLE_DEVICES=1 python examples/rlbench/run_rlbench_eval_batch.py  --model-name comeoncomeon --task_name close_jar  --port 8013  --use_reticle --reticle_config_key large_crosshair_dynamic_default_color


# greatlakes
source examples/rlbench/setup_greatlakes.sh 
DISPLAY=:9 /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-rlbench/bin/python examples/rlbench/run_rlbench_eval_batch.py  --model-name testestest --task_name close_jar  --port 8000 