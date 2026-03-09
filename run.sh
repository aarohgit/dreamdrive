#!/bin/bash

DATA_PATH=$1

apt-get update
apt-get install -y libgl1

pip install --upgrade pip
pip install -r requirements.txt

pip uninstall -y numpy
pip install numpy==1.26.4

cd submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation
cd ../../

cd submodules/simple-knn
pip install -e . --no-build-isolation
cd ../../

cd submodules/vista
pip install -e . --no-build-isolation
cd ../../

# Copy checkpoints
mkdir -p submodules/vista/ckpts
cp $DATA_PATH/checkpoints/* submodules/vista/ckpts/

# Copy images
mkdir -p submodules/vista/image_folder
cp -r $DATA_PATH/image_folder/* submodules/vista/image_folder/

mkdir -p data/benchmark

export PYTHONPATH=$PWD
export DATA_ROOT=submodules/vista/image_folder

python -m dreamdrive.diffusion.sample \
--n_frames 25 \
--n_rounds 5 \
--n_conds 5 \
--height 448 \
--width 768 \
--n_steps 50