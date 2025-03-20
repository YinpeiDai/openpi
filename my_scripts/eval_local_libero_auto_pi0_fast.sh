cd /home/daiyp/openpi


TASK_SUITE_NAME=libero_spatial,libero_object,libero_goal,libero_10
MODEL_NAME=pi0_fast_libero_large_crosshair_dynamic_default_color_grpsen_ckpt15000
DEVICE=0
PORT=8002
CKPT_DIR=/home/daiyp/openpi/runs/ckpts/pi0_fast_libero/pi0_fast_libero_large_crosshair_dynamic_default_color_grpsen/30000
USE_RETICLE=1
RETICLE_CFG=large_crosshair_dynamic_default_color
LEROBOT_REPO_ID=libero_large_crosshair_dynamic_default_color_grpsen
USE_GRSP_SENSE=1

SESSION_NAME="Eval-Pi0fast-${MODEL_NAME}-${TASK_SUITE_NAME}"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
  echo "Starting new tmux session: $SESSION_NAME"
  tmux new-session -d -s $SESSION_NAME
fi

sleep 2
tmux new-window -n server
tmux send-keys "source .venv/bin/activate" Enter
tmux send-keys "XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=${DEVICE} uv run scripts/serve_policy.py --port ${PORT}  --lerobot-repo-id ${LEROBOT_REPO_ID}  --no-apply-delta  policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=${CKPT_DIR}" Enter


sleep 2
tmux new-window -n client
tmux send-keys "source examples/libero/.venv/bin/activate" Enter
tmux send-keys "export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero" Enter
tmux send-keys "export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/scopereticle/src" Enter
# if use reticle, --use-reticle, else --no-use-reticle

if [ "$USE_RETICLE" = 1 ]; then
  if [ "$USE_GRSP_SENSE" = 1 ]; then
    tmux send-keys "MUJOCO_EGL_DEVICE_ID=${DEVICE} CUDA_VISIBLE_DEVICES=${DEVICE} /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT} --use_reticle --reticle_config_key ${RETICLE_CFG} --use_grasp_sense"  Enter
  else
    tmux send-keys "MUJOCO_EGL_DEVICE_ID=${DEVICE} CUDA_VISIBLE_DEVICES=${DEVICE} /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT} --use_reticle --reticle_config_key ${RETICLE_CFG}"  Enter
  fi
else
  tmux send-keys "MUJOCO_EGL_DEVICE_ID=${DEVICE} CUDA_VISIBLE_DEVICES=${DEVICE} /nfs/turbo/coe-chaijy/daiyp/micromamba_gl/envs/openpi-libero/bin/python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT}" Enter
fi