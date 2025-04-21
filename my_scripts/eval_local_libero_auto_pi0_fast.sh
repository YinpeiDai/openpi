cd /home/daiyp/openpi

# libero_spatial,libero_object,libero_goal,libero_10
TASK_SUITE_NAME=libero_object,libero_goal #,libero_object,libero_goal,libero_10
MODEL_NAME=lambda-pi0-fast-libero-large_crosshair_dynamic_default_color-20k
DEVICE=0
PORT=8002
CKPT_DIR=/home/daiyp/openpi/runs/ckpts/pi0_fast_libero/lambda-pi0-fast-libero-large_crosshair_dynamic_default_color-20k/20000
USE_RETICLE=1
RETICLE_CFG=large_crosshair_dynamic_default_color
LEROBOT_REPO_ID=large_crosshair_dynamic_default_color
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
    tmux send-keys "MUJOCO_EGL_DEVICE_ID=${DEVICE} CUDA_VISIBLE_DEVICES=${DEVICE} python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT} --use_reticle --reticle_config_key ${RETICLE_CFG} --use_grasp_sense"  Enter
  else
    tmux send-keys "MUJOCO_EGL_DEVICE_ID=${DEVICE} CUDA_VISIBLE_DEVICES=${DEVICE} python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT} --use_reticle --reticle_config_key ${RETICLE_CFG}"  Enter
  fi
else
  tmux send-keys "MUJOCO_EGL_DEVICE_ID=${DEVICE} CUDA_VISIBLE_DEVICES=${DEVICE} pythonexamples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT}" Enter
fi