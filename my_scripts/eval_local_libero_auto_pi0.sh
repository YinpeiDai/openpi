# Pi0 can not feed in half GPU

cd /home/daiyp/openpi

MODEL_NAME=pi0_ckpt30000
DEVICE=0
PORT=8001
CKPT_DIR=/nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/ckpts/pi0_libero/pi0_libero_finetune_bs32_acc2/30000
EVAL_TASK_SH=my_scripts/client_libero_10_spatial.sh
# EVAL_TASK_SH=my_scripts/client_libero_goal_object.sh 

SESSION_NAME="Eval-Pi0-${MODEL_NAME}"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
  echo "Starting new tmux session: $SESSION_NAME"
  tmux new-session -d -s $SESSION_NAME
fi

sleep 2
tmux new-window -n server
tmux send-keys "source .venv/bin/activate" Enter
tmux send-keys "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=${DEVICE} uv run scripts/serve_policy.py --port ${PORT}  policy:checkpoint --policy.config=pi0_libero --policy.dir=${CKPT_DIR}" Enter


sleep 2
tmux new-window -n client
tmux send-keys "bash ${EVAL_TASK_SH} ${PORT} ${DEVICE} ${MODEL_NAME}" Enter