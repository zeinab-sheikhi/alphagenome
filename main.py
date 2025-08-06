import torch.nn as nn
import torch
import torch.nn.functional as F


class StandardizedConv1D(nn.module):
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
        assert x.size(-1) == self.input_channels, f"number of input channels must be {x.size(-1)}"
        
        weight_mean = self.weight.mean(dim=(1, 2), keepdim=True)
        weight_var = self.weight.var(dim=(1, 2), keepdim=True)
        normalized_weight = self.weight - weight_mean / torch.sqrt(weight_var + self.eps)

        standard_weight = self.gain.view(-1, 1, 1) * normalized_weight

        output = F.conv1d(
            input=x.transpose(1, 2), 
            weight=standard_weight, 
            padding=self.kernel_size // 2, 
        )

        return output.transpose(1, 2)
