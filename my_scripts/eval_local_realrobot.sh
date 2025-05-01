# server
source .venv/bin/activate

# deploy the policy

# pi0-fast
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8012  --lerobot-repo-id realrobot_pickpalce_10demos_reticle  policy:checkpoint --policy.config=pi0_fast_realrobot --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_realrobot/pi0-realrobot_pickpalce_10demos_reticle/10000


# pi0
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8001  --lerobot-repo-id realrobot_pickpalce_10demos_reticle  policy:checkpoint --policy.config=pi0_realrobot --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_realrobot/pi0-realrobot_pickpalce_10demos_reticle/10000

XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port 8001  --lerobot-repo-id realrobot_ball_in_drawer_reticle  policy:checkpoint --policy.config=pi0_realrobot --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_realrobot/lambda-pi0-realrobot_ball_in_drawer_reticle/19999


XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8002  --lerobot-repo-id realrobot_egg_in_carton_reticle  policy:checkpoint --policy.config=pi0_realrobot --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_realrobot/lambda-pi0-realrobot_egg_in_carton_reticle/19999



XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/serve_policy.py --port 8008  --lerobot-repo-id realrobot_tracevla  policy:checkpoint --policy.config=pi0_realrobot --policy.dir=/home/ubuntu/chailab/daiyp/openpi/runs/ckpts/pi0_realrobot/lambda-pi0-realrobot_tracevla/30000
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port 8001  --lerobot-repo-id realrobot_bread_in_toaster_reticle  policy:checkpoint --policy.config=pi0_realrobot_long_horizon --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_realrobot_long_horizon/lambda-pi0-realrobot_bread_in_toaster_reticle_long_horizon/19999


XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port 8002  --lerobot-repo-id realrobot_egg_in_carton_reticle  policy:checkpoint --policy.config=pi0_realrobot_long_horizon --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_realrobot_long_horizon/lambda-pi0-realrobot_egg_in_carton_reticle_long_horizon/19999


XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8002 --lerobot-repo-id realrobot_all_tasks  policy:checkpoint --policy.config=pi0_realrobot --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_realrobot/lambda-pi0-realrobot_all_tasks/49999


XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port 8001 --lerobot-repo-id realrobot_all_tasks  policy:checkpoint --policy.config=pi0_fast_realrobot --policy.dir=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_fast_realrobot/lambda-pi0-fast-realrobot_all_tasks/49999
