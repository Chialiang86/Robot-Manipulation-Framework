# bin/sh

# Install Dependencies - Linux
sudo apt install libeigen3-dev libglfw3-dev libomp-dev libxinerama-dev libxcursor-dev libxi-dev git-lfs cmake libboost-all-dev
# Get the source code
git clone https://github.com/StrayRobots/3d-annotation-tool.git  
cd 3d-annotation-tool
# Initialize git submodules
git submodule update --init --recursive 
mkdir build
# Pull git-lfs objects (helper meshes etc)
it lfs install && git lfs pull 
# -j8 specifies the number of parallel jobs, for a fewer jobs use a lower number (than 8))
cd build && cmake .. && make -j8 