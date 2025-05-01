PORT=$1
KEY="large_crosshair_dynamic_default_color_new-rerun"
USE_GRSP_SENSE=0
MODEL_TYPE=$2
MODEL_TYPE_STR=$3
MODEL_NAME=${MODEL_TYPE}-libero-${KEY}-ckpt25k-public
CKPT_DIR=/home/ubuntu/chailab/daiyp/openpi/runs/ckpts/${MODEL_TYPE_STR}_libero/final-${MODEL_TYPE}-libero-${KEY}/25000
USE_RETICLE=1
RETICLE_CFG=large_crosshair_dynamic_default_color
LEROBOT_REPO_ID=large_crosshair_dynamic_default_color_new
TASK_SUITE_NAME=$4 #libero_goal
# libero_spatial,libero_object,libero_goal,libero_10 
GPU_FRACTION=$5

SESSION_NAME="Eval-${MODEL_NAME}-${TASK_SUITE_NAME}"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
  echo "Starting new tmux session: $SESSION_NAME"
  tmux new-session -d -s $SESSION_NAME
fi


sleep 1
tmux new-window -n server
tmux send-keys "source /home/ubuntu/chailab/daiyp/mybashrc.sh" Enter
tmux send-keys "XLA_PYTHON_CLIENT_MEM_FRACTION=${GPU_FRACTION} /home/ubuntu/chailab/daiyp/micromamba/envs/openpi/bin/python scripts/serve_policy.py --port ${PORT}  --lerobot-repo-id ${LEROBOT_REPO_ID}  --no-apply-delta  policy:checkpoint --policy.config=${MODEL_TYPE_STR}_libero --policy.dir=${CKPT_DIR}" Enter


sleep 1
tmux new-window -n client
tmux send-keys "source /home/ubuntu/chailab/daiyp/mybashrc.sh" Enter
tmux send-keys "source setup_libero.sh" Enter
# if use reticle, --use-reticle, else --no-use-reticle
if [ "$USE_RETICLE" = 1 ]; then
  if [ "$USE_GRSP_SENSE" = 1 ]; then
    tmux send-keys "python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT} --use_reticle --reticle_config_key ${RETICLE_CFG} --use_grasp_sense"  Enter
  else
    tmux send-keys "python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT} --use_reticle --reticle_config_key ${RETICLE_CFG}"  Enter
  fi
else
  tmux send-keys "python examples/libero/run_libero_eval_batch.py  --model-name ${MODEL_NAME} --task_suite_name ${TASK_SUITE_NAME} --port ${PORT} --task_start_id 0 --task_end_id 10" Enter
fi



