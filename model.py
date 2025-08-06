import torch
import torch.nn as nn 


class RMSBatchNorm1D(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6, momentum: float = 0.9):
        super().__init__()
        self.eps = eps 
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_var", torch.ones((num_features)))
    
    def forward(self, x: torch.Tensor):
        if self.training:
            if x.dim() == 2:
                # x.shape = (batch_size, num_features)
                batch_variance = torch.var(x, dim=0, unbiased=False)  # (num_features, )
            else:
                # x.shape = (batch_size, seq_len, num_features)
                x_reshaped = x.view(-1, x.size(-1))  # (batch_size * seq_len, num_features)
                batch_variance = torch.var(x_reshaped, dim=0, unbiased=False)  # (num_features, )
        else:
            batch_variance = self.running_var
        
        xhat = x / torch.sqrt(batch_variance + self.eps)  # (batch_size, seq_len, num_features)
        output = self.gamma * xhat + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_variance.detach()  # (num_features, )
       
        return output
