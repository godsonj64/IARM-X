import math
import torch
from torch import nn
import torch.nn.functional as F

from ..config import IARMXConfig
from .layers import RMSNorm, SwiGLU, RotaryEmbedding
from .resonance import ResonanceTransform
from .state import AttentionLayerState


class ResonanceAttentionBlock(nn.Module):
    """Exact causal attention whose Q/K geometry is transformed by resonance operators.

    Historical importance is cached alongside K/V so prefill, chunked inference,
    and one-token decoding implement the same score function.
    """

    def __init__(self, cfg: IARMXConfig):
        super().__init__()
        d, h, dh = cfg.dim, cfg.n_heads, cfg.head_dim
        self.cfg = cfg
        self.norm1 = RMSNorm(d, cfg.rms_eps)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.q_res = ResonanceTransform(d, h, cfg.n_operators, cfg.operator_rank)
        self.k_res = ResonanceTransform(d, h, cfg.n_operators, cfg.operator_rank)
        self.rope = RotaryEmbedding(dh, cfg.max_seq_len, cfg.rope_theta)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.norm2 = RMSNorm(d, cfg.rms_eps)
        self.ffn = SwiGLU(d, cfg.ffn_hidden, cfg.dropout)
        # Non-zero initialization lets gradients reach importance predictors on
        # the first optimization step instead of being temporarily disconnected.
        self.importance_scale = nn.Parameter(torch.full((h,), cfg.importance_scale_init))

    def initial_state(self, batch: int, device, dtype) -> AttentionLayerState:
        h, dh = self.cfg.n_heads, self.cfg.head_dim
        empty = torch.empty(batch, h, 0, dh, device=device, dtype=dtype)
        return AttentionLayerState(empty, empty.clone(), None)

    def _importance_all(
        self,
        state: AttentionLayerState | None,
        historical_importance: torch.Tensor | None,
        t: int,
        past: int,
        device,
        dtype,
    ) -> torch.Tensor | None:
        if not self.cfg.attention_bias_from_importance:
            return None
        cached = state.importance if state is not None else None
        current = None
        if historical_importance is not None:
            if historical_importance.ndim != 3 or historical_importance.shape[-1] != self.cfg.n_heads:
                raise ValueError("historical_importance must have shape [B,T,H]")
            current = historical_importance.transpose(1, 2).to(device=device, dtype=dtype)
        if past > 0 and cached is not None:
            if current is None:
                current = torch.ones(cached.size(0), cached.size(1), t, device=device, dtype=dtype)
            return torch.cat([cached.to(dtype=dtype), current], dim=-1)
        if past > 0 and cached is None and current is not None:
            prefix = torch.ones(current.size(0), current.size(1), past, device=device, dtype=dtype)
            return torch.cat([prefix, current], dim=-1)
        return current

    def _attention(
        self,
        qh: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
        past: int,
        importance: torch.Tensor | None,
    ) -> torch.Tensor:
        b, h, t, d = qh.shape
        s = kh.size(2)
        dropout_p = self.cfg.dropout if self.training else 0.0
        key_bias = None
        if importance is not None:
            if importance.shape != (b, h, s):
                raise ValueError(f"importance shape {tuple(importance.shape)} != {(b, h, s)}")
            key_bias = self.importance_scale.view(1, h, 1) * torch.log(
                importance.clamp_min(1e-6)
            )

        if self.cfg.use_sdpa:
            # Exact low-overhead encoding of a key-only additive score bias into one
            # extra Q/K coordinate. With explicit SDPA scale 1/sqrt(d):
            #   [q,1]·[k,b*sqrt(d)]/sqrt(d) = q·k/sqrt(d) + b.
            # This retains the optimized SDPA path without constructing a dense
            # [B,H,T,S] importance mask. Values keep their original head dimension.
            q_attn, k_attn = qh, kh
            if key_bias is not None:
                q_extra = torch.ones((*qh.shape[:-1], 1), device=qh.device, dtype=qh.dtype)
                k_extra = (key_bias * math.sqrt(d)).unsqueeze(-1).to(kh.dtype)
                q_attn = torch.cat([qh, q_extra], dim=-1)
                k_attn = torch.cat([kh, k_extra], dim=-1)

            scale = 1.0 / math.sqrt(d)
            if past == 0:
                return F.scaled_dot_product_attention(
                    q_attn, k_attn, vh, dropout_p=dropout_p, is_causal=True, scale=scale
                )
            if t == 1:
                return F.scaled_dot_product_attention(
                    q_attn, k_attn, vh, dropout_p=dropout_p, is_causal=False, scale=scale
                )

            qi = torch.arange(t, device=qh.device)[:, None] + past
            kj = torch.arange(s, device=qh.device)[None, :]
            causal = kj <= qi
            return F.scaled_dot_product_attention(
                q_attn, k_attn, vh, attn_mask=causal, dropout_p=dropout_p,
                is_causal=False, scale=scale
            )

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.cfg.head_dim)
        qi = torch.arange(t, device=qh.device)[:, None] + past
        kj = torch.arange(s, device=qh.device)[None, :]
        scores = scores.masked_fill(kj > qi, float("-inf"))
        if key_bias is not None:
            scores = scores + key_bias.unsqueeze(2).to(scores.dtype)
        attn = torch.softmax(scores.float(), dim=-1).to(qh.dtype)
        return torch.matmul(attn, vh)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        state: AttentionLayerState | None = None,
        historical_importance: torch.Tensor | None = None,
    ):
        b, t, d = x.shape
        hnorm = self.norm1(x)
        q = self.q_proj(hnorm).view(b, t, self.cfg.n_heads, self.cfg.head_dim)
        k = self.k_proj(hnorm).view_as(q)
        v = self.v_proj(hnorm).view_as(q)
        cos, sin = self.rope.cos_sin(positions, q.dtype)
        q = self.rope.apply_rotary(q, cos, sin)
        k = self.rope.apply_rotary(k, cos, sin)
        q, qg, _ = self.q_res(q, hnorm)
        k, kg, _ = self.k_res(k, hnorm)

        qh, kh, vh = (z.transpose(1, 2) for z in (q, k, v))
        if state is not None and state.key.numel() > 0:
            kh_all = torch.cat([state.key.to(kh.dtype), kh], dim=2)
            vh_all = torch.cat([state.value.to(vh.dtype), vh], dim=2)
            past = state.key.size(2)
        else:
            kh_all, vh_all, past = kh, vh, 0

        importance_all = self._importance_all(
            state, historical_importance, t, past, x.device, qh.dtype
        )
        out = self._attention(qh, kh_all, vh_all, past, importance_all)

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop(self.out_proj(out))
        x = x + self.drop(self.ffn(self.norm2(x)))
        new_state = AttentionLayerState(
            kh_all.detach(),
            vh_all.detach(),
            importance_all.detach() if importance_all is not None else None,
        )
        return x, new_state, {"q_gate": qg, "k_gate": kg}
