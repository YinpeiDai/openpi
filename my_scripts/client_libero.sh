# get the first args

# $1: the port number
# $2: the device number
# $3: the model name
# $4: task suite name, comma separated, e.g., libero_spatial,libero_10
# $5: whether to use reticle
# $6: the reticle config key


source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/scopereticle/src

# for each task suite, run the evaluation
IFS=',' read -r -a task_suites <<< "$4"

USE_RETICLE=$5

if [ "$USE_RETICLE" = 1 ]; then
    for task_suite in "${task_suites[@]}"
    do 
        CUDA_VISIBLE_DEVICES=$2 python examples/libero/run_libero_eval_batch.py  --model-name $3 --task_suite_name $task_suite --port $1 --use_reticle --reticle_config_key $6
        sleep 3
    done
else
    for task_suite in "${task_suites[@]}"
    do
        CUDA_VISIBLE_DEVICES=$2 python examples/libero/run_libero_eval_batch.py  --model-name $3 --task_suite_name $task_suite --port $1
        sleep 3
    done
fi
