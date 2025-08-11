import torch 


def geomspace(start, end, steps, device=None, dtype=torch.float32):
    """
    Generate `steps` points between `start` and `end` (inclusive) spaced geometrically.
    Equivalent to numpy.geomspace.
    """

    log_start = torch.log(torch.tensor(start, device=device, dtype=dtype))
    log_end = torch.log(torch.tensor(end, device=device, dtype=dtype))
    return torch.exp(torch.linspace(log_start, log_end, steps))


def apply_rope(
    x: torch.Tensor, 
    max_position: int, 
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    
    device = x.device
    seq_len = x.size(-2)
    dim = x.size(-1)
    assert dim % 2 == 0, "RoPE requires even last dimension"

    if positions is None:
        positions = torch.arange(seq_len, device=device) 
    
    num_freqs = dim // 2

    geomspace_vals = geomspace(1, max_position - num_freqs + 1, num_freqs, device=device)
    inv_freq = 1.0 / (torch.arange(num_freqs, device=device) + geomspace_vals)

    theta = torch.einsum("s,f->sf", positions, inv_freq)  # (seq_len, num_freqs)
    theta = theta.repeat_interleave(2, dim=-1)  # (seq_len, num_freqs * 2)

    x_even = x[..., :dim:2]
    x_odd = x[..., 1::2]
    x_rotated = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

    return x * torch.cos(theta) + x_rotated * torch.sin(theta)
