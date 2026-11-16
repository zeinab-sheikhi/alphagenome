"""All the Normalization layers preserve the input shape."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, n_channels: int, epsilon: float = 1e-6):
        super().__init__()
        self.n_channels = n_channels
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average squared input over the last dimension (channels)."""
        C = x.size(-1)
        if self.n_channels != C:
            raise ValueError(f"scale requires {self.n_channels} channels but {C} were given!")
        rms = (x.float().square().mean(dim=-1, keepdim=True) + self.epsilon) ** 0.5
        return self.scale * (x / rms)
        

class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RMSBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()


if __name__ == "__main__":
    x = torch.tensor([[[3, 4]], [[0, 2]]])
    
    n_channels = 2
    rms_norm = RMSNorm(n_channels)
    print(rms_norm(x))

