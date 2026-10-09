"""Sequence-parallel forms of the IARM-X recurrences.

Both functions are exact reformulations of the per-token loop in
``IARMXRecurrentBlock._scan_loop`` (equal in exact arithmetic; FP32 rounding
differs only by summation order). Everything runs in FP32 with autocast
disabled, so the persistent memories keep FP32 semantics under BF16 training.

Slow memory
-----------
``N_t = N_{t-1} + phi_t * v_t`` and ``D_t = D_{t-1} + phi_t`` are prefix sums.

Fast memory
-----------
Per head, with unit-norm key ``k_t`` (rank r), value ``v_t`` (head width d),
decay ``g_t``, erase ``e_t`` and write ``w_t``:

    S_t = g_t S_{t-1} - e_t k_t (k_t^T S_{t-1}) + w_t k_t v_t^T
        = g_t S_{t-1} + k_t delta_t^T,
    delta_t = w_t v_t - e_t S_{t-1}^T k_t,
    o_t = S_t^T q_t.

This is a gated delta rule whose write strength is decoupled from its erase
strength. Inside a chunk of length C that starts from state S_0, let
``G_t = sum_{i<=t} log g_i`` (so ``Gamma_t = exp(G_t)``). Unrolling gives the
unit lower-triangular system

    (I + L) Delta = U - diag(e_t Gamma_{t-1}) K S_0,
    L_tj = e_t exp(G_{t-1} - G_j) (k_t . k_j)     for j < t,

the UT/WY transform of the Householder-like product prod_t (g_t I - e_t k_t k_t^T).
With ``U~ = (I+L)^{-1} U`` and ``W~ = (I+L)^{-1} diag(e Gamma_prev) K``,
both independent of S_0 and therefore computed for all chunks at once,

    Delta = U~ - W~ S_0,
    O     = diag(Gamma) Q S_0 + (M o Q K^T) Delta,     M_tj = exp(G_t - G_j), j <= t,
    S_C   = Gamma_C S_0 + (K o Gamma_C/Gamma)^T Delta.

Only the last three lines run sequentially, once per chunk, so the sequential
depth falls from T to T/C and the work becomes batched matrix products.

Stability: the transition g_t I - e_t k_t k_t^T has eigenvalues g_t (multiplicity
r-1) and g_t - e_t. For any g_t in [0, 1] and e_t in (0, 1) they lie in (-1, 1]
((-0.1, 1) with the default decay bounds), so the state is non-expansive and
every decay ratio used above is <= 1.
"""

import torch
import torch.nn.functional as F


def _no_autocast(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)


def slow_memory_parallel(
    phi: torch.Tensor,
    v: torch.Tensor,
    num0: torch.Tensor,
    den0: torch.Tensor,
    eps: float,
):
    """phi, v [B,T,H,D]; num0, den0 [B,H,D] -> (slow [B,T,H,D], num_T, den_T), all FP32."""
    with _no_autocast(phi.device):
        ph = phi.float()
        num = num0.float().unsqueeze(1) + torch.cumsum(ph * v.float(), dim=1)
        den = den0.float().unsqueeze(1) + torch.cumsum(ph, dim=1)
        slow = num / den.clamp_min(eps)
    return slow, num[:, -1], den[:, -1]


def fast_memory_chunked(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    state: torch.Tensor,
    chunk_size: int = 64,
):
    """Chunkwise-parallel fast associative memory.

    k, q [B,T,H,R] (unit norm); v [B,T,H,D]; decay, erase, write [B,T,H];
    state [B,H,R,D]. Returns (reads [B,T,H,D], final state [B,H,R,D]), FP32.
    """
    b, t, h, r = k.shape
    d = v.size(-1)
    c = max(1, min(int(chunk_size), t))
    n = -(-t // c)
    pad = n * c - t

    with _no_autocast(k.device):
        # [B,T,H,*] -> [B,H,T,*]
        kk = k.float().transpose(1, 2)
        qq = q.float().transpose(1, 2)
        u = (write.float().unsqueeze(-1) * v.float()).transpose(1, 2)
        # The clamp only matters for memory_decay_min=0, where log(0) would give NaN ratios.
        log_g = torch.log(decay.float().clamp_min(torch.finfo(torch.float32).tiny)).transpose(1, 2)
        e = erase.float().transpose(1, 2)
        if pad:
            # Identity steps: g=1, e=0, u=0, k=q=0 leave state and earlier reads unchanged.
            kk = F.pad(kk, (0, 0, 0, pad))
            qq = F.pad(qq, (0, 0, 0, pad))
            u = F.pad(u, (0, 0, 0, pad))
            log_g = F.pad(log_g, (0, pad))
            e = F.pad(e, (0, pad))

        kk = kk.reshape(b, h, n, c, r)
        qq = qq.reshape(b, h, n, c, r)
        u = u.reshape(b, h, n, c, d)
        log_g = log_g.reshape(b, h, n, c)
        e = e.reshape(b, h, n, c)

        g_incl = log_g.cumsum(-1)  # G_t
        g_prev = g_incl - log_g  # G_{t-1}
        idx = torch.arange(c, device=k.device)
        strict = idx[:, None] > idx[None, :]
        incl = idx[:, None] >= idx[None, :]
        # Mask before exp: above the diagonal G_t - G_j > 0 and could overflow.
        decay_strict = torch.exp(
            (g_prev.unsqueeze(-1) - g_incl.unsqueeze(-2)).masked_fill(~strict, float("-inf"))
        )
        decay_incl = torch.exp(
            (g_incl.unsqueeze(-1) - g_incl.unsqueeze(-2)).masked_fill(~incl, float("-inf"))
        )

        lower = e.unsqueeze(-1) * decay_strict * (kk @ kk.transpose(-1, -2))
        rhs = torch.cat([u, (e * torch.exp(g_prev)).unsqueeze(-1) * kk], dim=-1)
        # unitriangular=True solves (I + lower) X = rhs; lower has a zero diagonal.
        sol = torch.linalg.solve_triangular(lower, rhs, upper=False, unitriangular=True)
        u_t, w_t = sol[..., :d], sol[..., d:]

        qk = (qq @ kk.transpose(-1, -2)) * decay_incl
        gamma = torch.exp(g_incl).unsqueeze(-1)  # [B,H,N,C,1]
        gamma_last = torch.exp(g_incl[..., -1]).unsqueeze(-1).unsqueeze(-1)  # [B,H,N,1,1]
        k_to_end = kk * torch.exp(g_incl[..., -1:] - g_incl).unsqueeze(-1)  # [B,H,N,C,R]

        s = state.float()
        reads = []
        for i in range(n):
            delta = u_t[:, :, i] - w_t[:, :, i] @ s
            reads.append(gamma[:, :, i] * (qq[:, :, i] @ s) + qk[:, :, i] @ delta)
            s = gamma_last[:, :, i] * s + k_to_end[:, :, i].transpose(-1, -2) @ delta

        out = torch.stack(reads, dim=2).reshape(b, h, n * c, d)[:, :, :t].transpose(1, 2)
    return out, s
