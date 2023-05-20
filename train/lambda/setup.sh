# Install Anaconda
cd /tmp
curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash anaconda.sh
cd
# After installed
source ~/.bashrc

conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch

# Install mmocr
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet

git clone https://github.com/napatswift/mmocr.git
cd mmocr
pip install -v -e .

# Download the thvl and textdet-thvote datasets from GitHub.
wget https://github.com/napatswift/vote-count/releases/download/v0.0.1/vl+vc-textdet.tar.gz
wget https://github.com/napatswift/syTH-doc/releases/download/v0.0.1/textdet-thvote.tar.gz

DATA_DIR="data/det/"
# creates a directory called `data/det/`, if not exist
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
fi

# extracts dataset to the `data/det/` directory
tar xzf vl+vc-textdet.tar.gz -C $DATA_DIR
tar xzf textdet-thvote.tar.gz -C $DATA_DIR