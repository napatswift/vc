apt-get update
apt-get install -y python3-setuptools tmux

pip3 install openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.1.0"

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .

wget https://github.com/napatswift/vc/releases/download/table-det-v-1ki/table-det-elect66.tar.gz
tar -zxf table-det-elect66.tar.gz