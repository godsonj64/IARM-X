import torch
from torch import nn
import torch.nn.functional as F

from ..config import IARMXConfig
from .layers import RMSNorm, SwiGLU, RotaryEmbedding, CausalDepthwiseConv1d
from .resonance import ResonanceTransform
from .state import RecurrentLayerState


class IARMXRecurrentBlock(nn.Module):
    """IARM core + IARM-X selective associative memory.

    Slow memory follows IARM's coordinatewise normalized cumulative average.
    IARM-X adds a finite-rank query-addressable associative memory with learned
    decay, erase and write controls. Persistent numerical accumulators are kept
    in FP32 even when model weights/activations use BF16 or FP16.
    """

    def __init__(self, cfg: IARMXConfig):
        super().__init__()
        d, h, dh = cfg.dim, cfg.n_heads, cfg.head_dim
        r = cfg.fast_memory_rank
        self.cfg = cfg
        self.norm1 = RMSNorm(d, cfg.rms_eps)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.q_res = ResonanceTransform(d, h, cfg.n_operators, cfg.operator_rank)
        self.k_res = ResonanceTransform(d, h, cfg.n_operators, cfg.operator_rank)
        self.rope = RotaryEmbedding(dh, cfg.max_seq_len, cfg.rope_theta)
        self.local = CausalDepthwiseConv1d(d, cfg.local_kernel)

        self.mem_key = nn.Linear(dh, r, bias=False)
        self.mem_query = nn.Linear(dh, r, bias=False)
        self.decay = nn.Linear(d, h, bias=True)
        self.erase = nn.Linear(d, h, bias=True)
        self.write = nn.Linear(d, h, bias=True)
        self.importance = nn.Linear(d, h, bias=True)

        self.fusion = nn.Linear(d, h * 3, bias=True)
        self.out_gate = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.norm2 = RMSNorm(d, cfg.rms_eps)
        self.ffn = SwiGLU(d, cfg.ffn_hidden, cfg.dropout)

    def _decay(self, h: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.decay(h))
        lo, hi = self.cfg.memory_decay_min, self.cfg.memory_decay_max
        return lo + (hi - lo) * raw

    def initial_state(self, batch: int, device, dtype) -> RecurrentLayerState:
        h, dh, r, k = self.cfg.n_heads, self.cfg.head_dim, self.cfg.fast_memory_rank, self.cfg.local_kernel
        # Slow and fast persistent memories must remain FP32: cumulative BF16
        # additions plateau at long context and change the recurrence itself.
        return RecurrentLayerState(
            slow_num=torch.zeros(batch, h, dh, device=device, dtype=torch.float32),
            slow_den=torch.zeros(batch, h, dh, device=device, dtype=torch.float32),
            fast=torch.zeros(batch, h, r, dh, device=device, dtype=torch.float32),
            conv_cache=torch.zeros(batch, max(k - 1, 0), self.cfg.dim, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor, state: RecurrentLayerState | None = None):
        b, t, d = x.shape
        hnorm = self.norm1(x)
        q = self.q_proj(hnorm).view(b, t, self.cfg.n_heads, self.cfg.head_dim)
        k = self.k_proj(hnorm).view_as(q)
        v = self.v_proj(hnorm).view_as(q)
        cos, sin = self.rope.cos_sin(positions, q.dtype)
        q = self.rope.apply_rotary(q, cos, sin)
        k = self.rope.apply_rotary(k, cos, sin)
        q, qg, qdelta = self.q_res(q, hnorm)
        k, kg, _ = self.k_res(k, hnorm)

        phi = F.elu(q) + 1.0
        if state is None:
            state = self.initial_state(b, x.device, hnorm.dtype)
        local_raw, new_conv_cache = self.local.forward_with_cache(hnorm, state.conv_cache)
        local = local_raw.view_as(v)

        num, den, fast = state.slow_num.float(), state.slow_den.float(), state.fast.float()
        slow_seq, fast_seq, imp_seq = [], [], []

        # Exact recurrent semantics. A fused chunkwise CUDA/Triton scan remains a
        # performance optimization; this reference path is the correctness oracle.
        for i in range(t):
            ph = phi[:, i].float()
            vv = v[:, i].float()
            num = num + ph * vv
            den = den + ph
            slow = num / den.clamp_min(self.cfg.slow_memory_eps)

            kk = F.normalize(self.mem_key(k[:, i]).float(), dim=-1)
            qq = F.normalize(self.mem_query(q[:, i]).float(), dim=-1)
            pred = torch.einsum("bhr,bhrd->bhd", kk, fast)
            decay = self._decay(hnorm[:, i]).float().unsqueeze(-1).unsqueeze(-1)
            erase = torch.sigmoid(self.erase(hnorm[:, i])).float().unsqueeze(-1).unsqueeze(-1)
            write = torch.sigmoid(self.write(hnorm[:, i])).float().unsqueeze(-1).unsqueeze(-1)
            fast = (
                decay * fast
                - erase * torch.einsum("bhr,bhd->bhrd", kk, pred)
                + write * torch.einsum("bhr,bhd->bhrd", kk, vv)
            )
            fast_read = torch.einsum("bhr,bhrd->bhd", qq, fast)
            importance = torch.sigmoid(self.importance(hnorm[:, i]))

            slow_seq.append(slow.to(v.dtype))
            fast_seq.append(fast_read.to(v.dtype))
            imp_seq.append(importance)

        slow_all = torch.stack(slow_seq, dim=1)
        fast_all = torch.stack(fast_seq, dim=1)
        importance = torch.stack(imp_seq, dim=1)
        mix = self.fusion(hnorm).view(b, t, self.cfg.n_heads, 3).softmax(-1)
        y = mix[..., 0:1] * slow_all + mix[..., 1:2] * fast_all + mix[..., 2:3] * local
        y = y.reshape(b, t, d)
        y = torch.sigmoid(self.out_gate(hnorm)) * y
        x = x + self.drop(self.out_proj(y))
        x = x + self.drop(self.ffn(self.norm2(x)))

        new_state = RecurrentLayerState(
            num.detach(), den.detach(), fast.detach(), new_conv_cache.detach()
        )
        aux = {
            "importance": importance,
            "q_gate": qg,
            "k_gate": kg,
            "q_delta_ratio": qdelta.norm(dim=-1).mean() / q.norm(dim=-1).mean().clamp_min(1e-8),
        }
        return x, new_state, aux
