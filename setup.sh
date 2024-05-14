#!/bin/bash

# Standard Stuff
apt update
apt upgrade -y
apt install git -y
apt install pip -y
apt-get install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install python3.7 -y
apt install vim

# Pip PyTorch compatible with CUDA
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# Mamba install
pip install --upgrade pip
pip install packaging
mkdir tmp
cd tmp
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.2.0
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v1.1.1 # current latest version tag
MAMBA_FORCE_BUILD=TRUE pip install .
cd ..
cd ..
rm -rf tmp

# TabPFN Installations
pip install -r requirements.txt

