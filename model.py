import torch
import torch.nn as nn 
import torch.nn.functional as F


class RMSBatchNorm1D(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6, momentum: float = 0.9):
        super().__init__()
        self.eps = eps 
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_var", torch.ones(num_features))
    
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


class StandardizedConv1D(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_size: int,
            eps: float = 1e-6,
        ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size 
        self.eps = eps 

        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, kernel_size))
        self.gain = nn.Parameter(torch.ones(output_channels))
    
    def forward(self, x: torch.Tensor):
        if x.size(-1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, but got {x.size(-1)}")
        
        weight_mean = torch.mean(self.weight, dim=(1, 2), keepdim=True)
        weight_var = torch.var(self.weight, dim=(1, 2), keepdim=True)
        weight_standardized = (self.weight - weight_mean) / torch.sqrt(weight_var + self.eps)
        scaled_weight = self.gain.view(-1, 1, 1) * weight_standardized

        # x shape: (batch_size, seq_len, input_channels)
        # conv1d expects: (batch_size, input_channels, seq_len)
        x_conv = x.transpose(1, 2)  # (batch_size, input_channels, seq_len)
        
        output = F.conv1d(
            input=x_conv,
            weight=scaled_weight,
            padding=self.kernel_size // 2
        )  # (batch_size, output_channels, seq_len)
        
        return output.transpose(1, 2)  # (batch_size, seq_len, output_channels)


class ConvBlock(nn.Module):
    def __init__(self, num_channels: int, width: int = 5):
        super().__init__()
        self.num_channels = num_channels
        self.width = width
    
    def forward(self, x: torch.Tensor):
        x = RMSBatchNorm1D(self.num_channels)(x)
        x = nn.GELU(x)
        if self.width == 1:
            x = nn.Linear(self.num_channels, self.num_channels)(x)
        else:
            x = 
