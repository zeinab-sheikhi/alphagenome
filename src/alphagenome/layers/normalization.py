"""All the Normalization layers preserve the input shape."""

import torch
from torch import nn


class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RMSNorm(nn.Module):
    """Each position across channels get normalized."""
    def __init__(self, n_channels: int, epsilon: float = 1e-6, learn_scale: bool = False):
        super().__init__()
        self.epsilon = epsilon
        self.n_channels = n_channels
        self.scale = nn.Parameter(torch.ones(n_channels)) if learn_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average squared input over the last dimension (channels)."""
        C = x.size(-1)
        if self.n_channels != C:
            raise ValueError(f"scale requires {self.n_channels} channels but {C} were given!")
        rms = (x.float().square().mean(dim=-1, keepdim=True) + self.epsilon) ** 0.5
        output = self.scale * (x / rms) if self.scale is not None else (x / rms)
        return output


class RMSBatchNorm(nn.Module):
    """Each channel across batch and positions get normalized"""
    def __init__(self, n_channels: int, epsilon: float = 1e-6, decay: float = 0.9):
        super().__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.n_channels = n_channels
        self.scale = nn.Parameter(torch.ones(n_channels))
        self.offset = nn.Parameter(torch.zeros(n_channels))
        self.register_buffer("running_mean_square", torch.ones(n_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(-1)
        if C != self.n_channels:
            raise ValueError(f"scale requires {self.n_channels} channels but {C} were given!")
        mean_square = x.float().square().mean(dim=(0, 1), keepdim=True)
        if self.training:
            self.running_mean_square = self.decay * self.running_mean_square + (1 - self.decay) * mean_square 
        
        return (self.scale * (x / self.running_mean_square)) + self.offset


if __name__ == "__main__":
    x = torch.tensor([[[3, 4]], [[0, 2]]])
    
    n_channels = 2
    rms_norm = RMSNorm(n_channels)
    rms_batch_norm = RMSBatchNorm(n_batch=2)
    print(rms_norm(x))
    print(rms_batch_norm(x))



class RMSNorm(nn.Module):
    def __init__(self, n_channels: int, epsilon: float = 1e-6, scale_active: bool = False):
        self.n_channels = n_channels
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(n_channels)) if scale_active else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(-1)
        if C != self.n_channels:
            raise ValueError("")

        rms = x / (x.float().square().mean(dim=-1, keepdim=True) + self.epsilon ) ** 0.5
        output = self.scale * rms if self.scale is not None else rms
        return output