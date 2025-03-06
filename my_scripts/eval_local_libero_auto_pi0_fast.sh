cd /home/daiyp/openpi

TASK_SUITE_NAME=libero_spatial
MODEL_NAME=ttttttt
DEVICE=0
PORT=8001
CKPT_DIR=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_fast_libero/pi0_fast_libero_finetune_bs32/30000
USE_RETICLE=1
RETICLE_CFG=large_crosshair_dynamic_default_color

SESSION_NAME="Eval-Pi0fast-${MODEL_NAME}-${TASK_SUITE_NAME}"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
  echo "Starting new tmux session: $SESSION_NAME"
  tmux new-session -d -s $SESSION_NAME
fi

sleep 2
tmux new-window -n server
tmux send-keys "source .venv/bin/activate" Enter
tmux send-keys "XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=${DEVICE} uv run scripts/serve_policy.py --port ${PORT}  policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=${CKPT_DIR}" Enter


sleep 2
tmux new-window -n client
tmux send-keys "source examples/libero/.venv/bin/activate" Enter
tmux send-keys "export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero" Enter
tmux send-keys "export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/scopereticle/src" Enter
# if use reticle, --use-reticle, else --no-use-reticle
if [ "$USE_RETICLE" = 1 ]; then
  tmux send-keys "CUDA_VISIBLE_DEVICES=${DEVICE} python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT} --use_reticle --reticle_config_key ${RETICLE_CFG}" Enter
else
  tmux send-keys "CUDA_VISIBLE_DEVICES=${DEVICE} python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT}" Enter
fi