# source examples/libero/.venv/bin/activate
micromamba activate openpi-libero
export MUJOCO_GL="osmesa"
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/scopereticle/src
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# python examples/libero/main.py --args.task-suite-name libero_object