cd /home/daiyp/openpi


# stack_blocks,open_drawer
# stack_cups,light_bulb_in,meat_off_grill
# insert_onto_square_peg,close_jar,push_buttons

TASK_NAME=insert_onto_square_peg,close_jar,push_buttons
MODEL_NAME=pi0_fast_rlbench_ckpt20000
DEVICE=1
PORT=8003
CKPT_DIR=/home/daiyp/openpi/runs/ckpts/pi0_fast_libero/pi0_fast_rlbench/20000
USE_RETICLE=0
RETICLE_CFG=large_crosshair_dynamic_default_color
APPLY_DELTA=0
lerobot_repo_id=rlbench

SESSION_NAME="Eval-Pi0fast-${MODEL_NAME}-${TASK_NAME}"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
  echo "Starting new tmux session: $SESSION_NAME"
  tmux new-session -d -s $SESSION_NAME
fi

sleep 2
tmux new-window -n server
tmux send-keys "source .venv/bin/activate" Enter

if [ "$APPLY_DELTA" = 1 ]; then
  tmux send-keys "XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=${DEVICE} uv run scripts/serve_policy.py --port ${PORT} --lerobot_repo_id ${lerobot_repo_id} policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=${CKPT_DIR}" Enter
else
  tmux send-keys "XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=${DEVICE} uv run scripts/serve_policy.py --port ${PORT} --lerobot_repo_id ${lerobot_repo_id} --no-apply-delta  policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=${CKPT_DIR}" Enter
fi


sleep 2
tmux new-window -n client
tmux send-keys "source examples/rlbench/.venv/bin/activate" Enter
tmux send-keys "source examples/rlbench/setup_bash.sh" Enter

# if use reticle, --use-reticle, else --no-use-reticle
if [ "$USE_RETICLE" = 1 ]; then
  tmux send-keys "MUJOCO_EGL_DEVICE_ID=${DEVICE} CUDA_VISIBLE_DEVICES=${DEVICE} python examples/rlbench/run_rlbench_eval_batch.py  --model-name ${MODEL_NAME} --task-name ${TASK_NAME} --port ${PORT} --use_reticle --reticle_config_key ${RETICLE_CFG}" Enter
else
  tmux send-keys "MUJOCO_EGL_DEVICE_ID=${DEVICE} CUDA_VISIBLE_DEVICES=${DEVICE} python examples/rlbench/run_rlbench_eval_batch.py  --model-name ${MODEL_NAME} --task-name ${TASK_NAME} --port ${PORT}" Enter
fi