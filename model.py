import torch
import torch.nn as nn 
import torch.nn.functional as F


class RMSBatchNorm1D(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        momentum: float = 0.9,
    ):
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
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size 
        self.eps = eps 

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.gain = nn.Parameter(torch.ones(out_channels))
    
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_channels, ValueError(f"Expected {self.in_channels} input channels, but got {x.size(-1)}")
        
        weight_mean = torch.mean(self.weight, dim=(1, 2), keepdim=True)
        weight_var = torch.var(self.weight, dim=(1, 2), keepdim=True)
        weight_standardized = (self.weight - weight_mean) / torch.sqrt(weight_var + self.eps)
        scaled_weight = self.gain.view(-1, 1, 1) * weight_standardized

        x_conv = x.transpose(1, 2)  # (batch_size, input_channels, seq_len)
        
        output = F.conv1d(
            input=x_conv,
            weight=scaled_weight,
            padding=self.kernel_size // 2
        )
        
        return output.transpose(1, 2)  # (batch_size, seq_len, output_channels)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.rms_norm = RMSBatchNorm1D(num_features=out_channels)
        self.gelu = nn.GELU()
        self.linear = nn.Linear(in_channels, out_channels)
        self.conv = StandardizedConv1D(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
        )
    
    def forward(self, x: torch.Tensor):
        x = self.rms_norm(x)
        x = self.gelu(x)
        if self.kernel_size == 1:
            x = self.linear(x)
        else:
            x = self.conv(x)
        return x


class DNAEmbedder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 768,
        kernel_size: int = 15,
    ):
        super().__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2, 
        )
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        out = self.conv_layer(x)  # (batch_size, num_channels, seq_len)
        out = out.transpose(1, 2)
        return out + self.conv_block(out)


class DownResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size)
    
    def forward(self, x: torch.Tensor):
        out = self.conv1(x)

        # Skip connection with padding
        if x.size(-1) != out.size(-1):
            padding = torch.zeros(x.size(0), x.size(1), out.size(-1) - x.size(-1), device=x.device)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x
        
        out = out + x_padded
        out = out + self.conv2(out)
        return out
