# bin/sh

sudo apt update
sudo apt install libopencv-dev python3-opencv

python3 -m pip install numpy dash dash_bootstrap_components --user
git clone https://github.com/luiscarlosgph/keypoint-annotation-tool.git
cd keypoint-annotation-tool

python3 setup.py install --user