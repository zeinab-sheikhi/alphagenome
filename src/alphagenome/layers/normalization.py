"""All the Normalization layers preserve the input shape."""
import torch 
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BatchNorm(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RMSBatchNorm(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()


if __name__ == "__main__":
    x = torch.tensor([[[3, 4]], [[0, 2]]])
    print(x.shape)
