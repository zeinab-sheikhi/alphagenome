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

        self.register_buffer("running_rms_squared", torch.ones(num_features))
    
    def forward(self, x: torch.Tensor):
        if self.training:
            batch_rms_squared = torch.mean(x ** 2, dim=0, keepdim=True)  # (1, seq_len, num_features)
        else:
            batch_rms_squared = self.running_rms_squared.view(1, 1, -1)  # (1, 1, num_features)
        
        xhat = x / torch.sqrt(batch_rms_squared + self.eps)  # (batch_size, seq_len, num_features)
        output = self.gamma.view(1, 1, -1) * xhat + self.beta.view(1, 1, -1)
        
        if self.training:
            with torch.no_grad():
                current_rms_squared = torch.mean(x**2, dim=(0, 1))  # (num_features,)
                self.running_rms_squared = self.momentum * self.running_rms_squared + (1 - self.momentum) * current_rms_squared
       
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
        self.rms_norm = RMSBatchNorm1D(num_features=in_channels)
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


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        initial_channels: int = 768, 
        channel_increment: int = 128,
        bin_sizes: list = None,
    ):
        super().__init__()
        
        if bin_sizes is None:
            bin_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        self.bin_sizes = bin_sizes
        self.channel_increment = channel_increment

        self.blocks = nn.ModuleDict()
        self.max_pools = nn.ModuleDict()

        current_channels = in_channels
        for i, bin_size in enumerate(self.bin_sizes):
            
            # Use DNAEmbedder for bin_size = 1
            if bin_size == 1:
                self.blocks[f"bin_{bin_size}"] = DNAEmbedder(
                    in_channels=current_channels, 
                    out_channels=initial_channels,
                )
                current_channels = initial_channels
            else:
                out_channels = current_channels + channel_increment

                self.blocks[f"bin_{bin_size}"] = DownResBlock(
                    in_channels=current_channels, 
                    out_channels=out_channels,
                )
                current_channels = out_channels
            
            # Add max pooling (except for the last bin size)
            if i < len(self.bin_sizes) - 1:
                self.max_pools[f"pool_{bin_size}"] = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        intermediates = {}
        for i, bin_size in enumerate(self.bin_sizes):
            x = self.blocks[f"bin_{bin_size}"](x)
            intermediates[f"bin_size_{bin_size}"] = x

            if i < len(self.bin_sizes) - 1:
                x = x.transpose(1, 2)
                x = self.max_pools[f"pool_{bin_size}"](x)
                x = x.transpose(1, 2)
        
        return x, intermediates
