#!/bin/bash
set -e

echo "Updating system"
sudo apt-get update -y

echo "Installing uv"
pip install uv

echo "Creating virual environment"
uv venv vov --python 3.12.7

echo "Installing pip"
./vov/bin/python -m ensurepip --upgrade

echo "Install uv pip"
./vov/bin/python -m pip install uv

echo "Installing packages"
./vov/bin/python -m uv pip install torch torchvision matplotlib noise tqdm pygame pygame_screen_recorder json torch_geometric