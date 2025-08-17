#!/bin/bash
set -e

echo "Updating system"
sudo apt-get update -y

pip install uv

uv venv vov --python 3.12.7
# source vov/bin/activate
./vov/bin/python -m pip install uv
./vov/bin/python -m uv pip install torch torchvision matplotlib noise tqdm pygame pygame_screen_recorder json torch_geometric