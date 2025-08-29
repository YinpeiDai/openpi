# install openpi
cd /home/daiyp/openpi
pip -r requirement.txt
# install lerobot
micromamba install rerun-sdk==0.21.0
pip install git+https://github.com/huggingface/lerobot@6674e368249472c91382eb54bb8501c94c7f0c56

pip install -e .
pip install -e packages/openpi-client




# install openpi-libero
pip install -e packages/openpi-client
pip install -e third_party/libero

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

source examples/libero/.venv/bin/activate
# set symbolic link
ln -s /nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/VLA/openpi /home/daiyp/.cache/openpi

ln -s  /nfs/turbo/coe-chaijy-unreplicated/daiyp/openpi/data  /home/daiyp/openpi/runs