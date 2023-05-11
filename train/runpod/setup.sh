apt-get update
apt-get install -y python3-setuptools tmux

pip install -U openmim
mim install mmengine
mim install mmcv mmdet mmocr

git clone https://github.com/napatswift/mmocr.git
cd mmocr

# Download the thvl and textdet-thvote datasets from GitHub.
wget https://github.com/napatswift/vote-count/releases/download/v0.0.2/vl+vc-textdet.tar.gz
wget https://github.com/napatswift/syTH-doc/releases/download/v0.0.1/textdet-thvote.tar.gz

DATA_DIR="data/det/"
# creates a directory called `data/det/`, if not exist
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
fi

# extracts dataset to the `data/det/` directory
tar xzf vl+vc-textdet.tar.gz -C $DATA_DIR
tar xzf textdet-thvote.tar.gz -C $DATA_DIR