#!/bin/bash

# Standard Stuff
apt update
apt upgrade -y
apt install git -y
apt install pip -y
apt update
apt install vim -y

# Pip PyTorch compatible with CUDA
#pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# TabPFN Installations
pip install -r requirements.txt

pip install wandb

pip install causal-conv1d>=1.4.0

pip install mamba-ssm