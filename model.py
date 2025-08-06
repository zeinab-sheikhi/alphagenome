import torch
import torch.nn as nn 


class RMSBatchNorm1D(nn.Module):
    def __init__(self, num_features: int, eps: int = 1e-6, momentum: float = 0.9):
        super().__init__()
        self.training = True
        self.eps = eps 
        self.momentum = momentum

        self.gamma = nn.Parameter((num_features))
        self.beta = nn.Parameter((num_features))

        self.register_buffer("running_var", torch.ones((num_features)))
    
    def forward(self, x: torch.Tensor):
        if x.shape[1] == 2:
            batch_variance = torch.var(x, dim=0)
        else:
            x_reshaped = x.view(x.shape[0], -1)
            batch_variance = torch.var(x_reshaped, dim=0, unbiased=False)
        if self.training:
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_variance
        xhat = x / torch.sqrt(self.running_var + self.eps)
        return self.gamma * xhat + self.beta
