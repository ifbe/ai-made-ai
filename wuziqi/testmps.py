import torch

if __name__ == "__main__":
    print("mps=", torch.backends.mps.is_available(), sep='')
