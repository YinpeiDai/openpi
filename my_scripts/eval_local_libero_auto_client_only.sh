source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

CUDA_VISIBLE_DEVICES=0 python examples/libero/run_libero_eval_batch.py  --model-name pi0_ckpt30000 --task_suite_name libero_10 --task_start_id 6 --task_end_id 10 --port 8000
sleep 3
CUDA_VISIBLE_DEVICES=0 python examples/libero/run_libero_eval_batch.py  --model-name pi0_ckpt30000 --task_suite_name libero_object --task_start_id 0 --task_end_id 10 --port 8000
sleep 3
CUDA_VISIBLE_DEVICES=0 python examples/libero/run_libero_eval_batch.py  --model-name pi0_ckpt30000 --task_suite_name libero_spatial --task_start_id 0 --task_end_id 10 --port 8000
sleep 3