# get the first args

# $1: the path to the root directory of the project
# $2: the port number
# $3: the device number
# $4: the model name

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

CUDA_VISIBLE_DEVICES=$3 python examples/libero/run_libero_eval_batch.py  --model-name $4 --task_suite_name libero_10  --port $2
sleep 3
CUDA_VISIBLE_DEVICES=$3 python examples/libero/run_libero_eval_batch.py  --model-name $4 --task_suite_name libero_spatial --port $2
sleep 3
