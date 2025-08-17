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

echo "Upgrading pip"
./vov/bin/python -m pip install --upgrade pip

echo "Install uv pip"
./vov/bin/python -m pip install uv

echo "Installing packages"
./vov/bin/python -m uv pip install torch torchvision matplotlib noise tqdm pygame pygame_screen_recorder torch_geometric

echo "Installing gnngroups"
./vov/bin/python -m uv pip install -e .

echo "Verifying cuda installation"
./vov/bin/python -c "import torch; print(torch.cuda.is_available())"
