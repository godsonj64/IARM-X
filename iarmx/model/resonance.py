import math
import torch
from torch import nn


class ResonanceTransform(nn.Module):
    """Softmax-gated mixture of learned low-rank operators.

    This preserves the central IARM idea: each head carries O factorized operators
    U_o V_o^T and a data-dependent convex gate.
    """

    def __init__(self, dim: int, n_heads: int, n_operators: int, rank: int):
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must divide n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_operators = n_operators
        self.rank = rank
        self.gate = nn.Linear(dim, n_heads * n_operators, bias=False)
        self.u = nn.Parameter(torch.empty(n_heads, n_operators, self.head_dim, rank))
        self.v = nn.Parameter(torch.empty(n_heads, n_operators, self.head_dim, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.u, std=0.02)
        nn.init.normal_(self.v, std=0.02)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, x: torch.Tensor, controller: torch.Tensor):
        # x [B,T,H,Dh], controller [B,T,D]
        b, t, h, d = x.shape
        gates = self.gate(controller).view(b, t, h, self.n_operators).softmax(-1)
        # V^T x -> [B,T,H,O,R]
        low = torch.einsum("bthd,hodr->bthor", x, self.v)
        # U (V^T x) -> [B,T,H,O,D]
        resp = torch.einsum("bthor,hodr->bthod", low, self.u)
        delta = (gates.unsqueeze(-1) * resp).sum(-2) / math.sqrt(self.rank)
        return x + delta, gates, delta
