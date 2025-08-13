import torch
import torch.nn as nn 
import torch.nn.functional as F

from typing import Optional, Tuple
from alphagenome.utils import apply_rope


class RMSBatchNorm1D(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        momentum: float = 0.9,
        device: torch.device | None = None
    ):
        super().__init__()
        self.eps = eps 
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features, device=device, dtype=torch.float32))  # scale
        self.beta = nn.Parameter(torch.zeros(num_features, device=device, dtype=torch.float32))  # shift

        self.register_buffer("running_rms_squared", torch.ones(num_features, device=device, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        if self.training:
            batch_rms_squared = torch.mean(x**2, dim=0, keepdim=True)  # (1, seq_len, num_features)
        else:
            batch_rms_squared = self.running_rms_squared.view(1, 1, -1)  # (1, 1, num_features)
        
        xhat = x * torch.rsqrt(batch_rms_squared + self.eps)  # (batch_size, seq_len, num_features)
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
        
        weight_mean = torch.mean(self.weight, dim=(1, 2), keepdim=True)  # (out_channels, 1, 1)
        weight_var = torch.var(self.weight, dim=(1, 2), keepdim=True)  # (out_channels, 1, 1)
        weight_standardized = (self.weight - weight_mean) / torch.sqrt(weight_var + self.eps)
        scaled_weight = self.gain.view(-1, 1, 1) * weight_standardized

        x_conv = x.transpose(1, 2)  # (batch_size, in_channels, seq_len)
        
        output = F.conv1d(
            input=x_conv,
            weight=scaled_weight,
            padding=self.kernel_size // 2
        )
        
        return output.transpose(1, 2)  # (batch_size, seq_len, out_channels)


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
        )
        self.conv_block = ConvBlock(in_channels=out_channels, out_channels=out_channels)
    
    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        out = self.conv_layer(x)  # (batch_size, out_channels, seq_len)
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
        return out + self.conv2(out)


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        initial_channels: int = 768, 
        channel_increment: int = 128,
        bin_sizes: Optional[list] = None,
    ):
        super().__init__()
        
        if bin_sizes is None:
            bin_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        self.bin_sizes = bin_sizes
        self.channel_increment = channel_increment

        self.blocks = nn.ModuleDict()
        self.max_pools = nn.ModuleDict()

        current_channels = in_channels
        
        for bin_size in self.bin_sizes:    
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
            
            self.max_pools[f"pool_{bin_size}"] = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        intermediates = {}
        for bin_size in self.bin_sizes:
            x = self.blocks[f"bin_{bin_size}"](x)
            intermediates[f"bin_size_{bin_size}"] = x
            x = x.transpose(1, 2)
            x = self.max_pools[f"pool_{bin_size}"](x)
            x = x.transpose(1, 2)
        
        return x, intermediates


class RMSBatchNormPairwise(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.norm = RMSBatchNorm1D(input_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, P_, F_ = x.shape
        x_reshaped = x.view(B, P * P_, F_)
        x_norm = self.norm(x_reshaped)
        return x_norm.view(B, P, P_, F_)


class LinearPairwise(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, P_, F_ = x.shape
        x_reshaped = x.view(B, P * P_, F_)
        x_proj = self.linear(x_reshaped)
        return x_proj.view(B, P, P_, -1)


class AttentionBiasBlock(nn.Module):
    def __init__(
        self, 
        num_pairwise_channels: int,
        num_heads: int = 8, 
        repeat_factor: int = 16, 
    ):
        super().__init__()
        self.num_heads = num_heads 
        self.repeat_factor = repeat_factor 

        self.sequential = nn.Sequential(
            RMSBatchNormPairwise(num_pairwise_channels), 
            nn.GELU(), 
            LinearPairwise(num_pairwise_channels, num_heads),
        )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequential(x)  # (B, P, P, H)

        # Repeat and permute 
        x = x.repeat_interleave(self.repeat_factor, dim=1).repeat_interleave(self.repeat_factor, dim=2)  # (B, S, S, H)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, S, S)
        return x
        

class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_heads: int = 8, 
        q_dim: int = 128, 
        k_dim: int = 128, 
        v_dim: int = 192,
        dropout: float = 0.1,
        rope_max_pos: int = 8192,
    ):
        super().__init__()

        self.h = num_heads
        self.dq = q_dim
        self.dk = k_dim
        self.dv = v_dim
        self.rope_max_pos = rope_max_pos

        self.pre_norm = RMSBatchNorm1D(input_dim)
        
        # Multi-query attention: 8 query heads, 1 shared key/value head
        self.q_proj = nn.Linear(input_dim, num_heads * q_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, k_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, v_dim, bias=False)

        # LayerNorm on per-head channels
        self.q_layernorm = nn.LayerNorm(q_dim)
        self.k_layernorm = nn.LayerNorm(k_dim)
        self.v_layernorm = nn.LayerNorm(v_dim)

        self.out_proj = nn.Linear(num_heads * v_dim, input_dim, bias=True)
        self.post_norm = RMSBatchNorm1D(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
    
        B, S, _ = x.shape
        x = self.pre_norm(x)
        
        # Projections
        q = self.q_proj(x)  # (B, S, H * Q)
        k = self.k_proj(x)  # (B, S, K)
        v = self.v_proj(x)  # (B, S, V)

        # Headify Q
        q = q.view(B, S, self.h, self.dq).permute(0, 2, 1, 3).contiguous()  # (B, H, S, Q)
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)
        v = self.v_layernorm(v)

        # RoPE on Q and K only
        q = q.view(B * self.h, S, self.dq)
        q = apply_rope(q, max_position=self.rope_max_pos)
        q = q.view(B, self.h, S, self.dq)

        k = apply_rope(k, max_position=self.rope_max_pos)

        # Attention logits
        att_logits = torch.einsum("bhid,bjd->bhij", q, k) * self.dk ** -0.5
        if attention_bias is not None:
            att_logits = att_logits + attention_bias  # (B, H, S, S)
        
        # soft-clip logits in [-5, 5]
        att_logits = torch.tanh(att_logits / 5.0) * 5.0

        attn = F.softmax(att_logits, dim=-1)
        
        y = torch.einsum("bhij, bjd->bhid", attn, v)  # (B, H, S, V)

        # Reshape and project back to input dimension
        y = y.permute(0, 2, 1, 3).contiguous().view(B, S, self.h * self.dv)
        y = self.out_proj(y)  # (B, S, C)
        y = self.post_norm(y)
        y = self.dropout(y)
        return y


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            RMSBatchNorm1D(input_dim), 
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * input_dim, input_dim), 
            RMSBatchNorm1D(input_dim),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,   # C - sequence feature dimension
        pair_dim: int,  # F - pairwise feature dimension
        num_layers: int = 9, 
        num_heads: int = 8, 
        repeat_factor: int = 16, 
        q_dim: int = 128, 
        k_dim: int = 128, 
        v_dim: int = 192, 
        dropout_rate: float = 0.1, 
    ):
        super().__init__()
        self.num_layers = num_layers

        self.pair_update_block = None

        self.attention_bias_blocks = nn.ModuleList([
            AttentionBiasBlock(pair_dim, num_heads, repeat_factor)
            for _ in range(num_layers)
        ])

        self.mha_blocks = nn.ModuleList([
            MultiHeadAttentionBlock(input_dim, num_heads, q_dim, k_dim, v_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.mlp_blocks = nn.ModuleList([
            MLPBlock(input_dim, dropout_rate)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,  # (B, S, C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_x = None

        for i in range(self.num_layers):
            # Update pairwise features on even layers
            if i % 2 == 0:
                pair_x = self.pair_update_block(x, pair_x)
            
            attention_bias = self.attention_bias_blocks[i](pair_x)
            x = x + self.mha_blocks[i](x, attention_bias)
            x = x + self.mlp_blocks[i](x)

        return x, pair_x


class SequenceToPairBlock(nn.Module):
    def __init__(self, in_din: int, out_dim: int = 512, num_heads: int = 32, k: int = 128):

        super().__init__()
        avg_pooling = nn.AdaptiveAvgPool1d(out_dim)
        rms_norm = nn.RMSNorm(out_dim)
        self.q_proj = nn.Linear(num_heads, k, bias=False)
        self.k_proj = nn.Linear(num_heads, k, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.avg_pooling(x)
        x = x.permute(0, 2, 1)
        x = self.rms_norm(x)
        
