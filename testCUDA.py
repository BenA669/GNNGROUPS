import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use GPU.")
else:
    print("CUDA is not available. Using CPU.")