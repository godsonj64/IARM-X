import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(F.silu(self.gate(x)) * self.up(x)))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def cos_sin(self, positions: torch.Tensor, dtype: torch.dtype):
        freqs = torch.einsum("...t,d->...td", positions.float(), self.inv_freq)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x [B,T,H,D], cos/sin [T,D/2] or [B,T,D/2]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.ndim == 3:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal convolution with an exact rolling inference cache."""

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> causal left padding only.
        y = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return self.conv(y).transpose(1, 2)

    def forward_with_cache(self, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same causal convolution using exactly K-1 cached normalized inputs.

        cache has shape [B,K-1,D]. A zero cache is mathematically identical to the
        left-zero-padding used by ``forward`` during an initial prefill.
        """
        k = self.kernel_size
        if k == 1:
            y = self.conv(x.transpose(1, 2)).transpose(1, 2)
            return y, x[:, :0]
        expected = (x.size(0), k - 1, x.size(2))
        if tuple(cache.shape) != expected:
            raise ValueError(f"conv cache shape {tuple(cache.shape)} != expected {expected}")
        history = torch.cat([cache.to(dtype=x.dtype), x], dim=1)
        y = self.conv(history.transpose(1, 2)).transpose(1, 2)
        new_cache = history[:, -(k - 1):].detach()
        return y, new_cache
