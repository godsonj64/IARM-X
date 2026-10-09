import math

import pytest
import torch

from iarmx import IARMXConfig, IARMXForCausalLM
from iarmx.model.resonance import ResonanceTransform
from iarmx.model.scan import fast_memory_chunked, slow_memory_parallel


def reference_fast(k, q, v, decay, erase, write, s):
    reads = []
    for i in range(k.size(1)):
        kk, qq = k[:, i], q[:, i]
        pred = torch.einsum("bhr,bhrd->bhd", kk, s)
        s = (
            decay[:, i, :, None, None] * s
            - erase[:, i, :, None, None] * torch.einsum("bhr,bhd->bhrd", kk, pred)
            + write[:, i, :, None, None] * torch.einsum("bhr,bhd->bhrd", kk, v[:, i])
        )
        reads.append(torch.einsum("bhr,bhrd->bhd", qq, s))
    return torch.stack(reads, dim=1), s


def random_inputs(b=2, t=37, h=3, r=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(b, t, h, r, generator=g), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(b, t, h, r, generator=g), dim=-1)
    v = torch.randn(b, t, h, d, generator=g)
    decay = 0.9 + 0.0999 * torch.rand(b, t, h, generator=g)
    erase = torch.rand(b, t, h, generator=g)
    write = torch.rand(b, t, h, generator=g)
    s0 = 0.5 * torch.randn(b, h, r, d, generator=g)
    return k, q, v, decay, erase, write, s0


@pytest.mark.parametrize("chunk", [1, 3, 8, 16, 64])
def test_fast_memory_chunked_matches_per_token_recurrence(chunk):
    inputs = [x.requires_grad_() for x in random_inputs()]
    ref_out, ref_s = reference_fast(*inputs)
    out, s = fast_memory_chunked(*inputs, chunk_size=chunk)
    assert torch.allclose(out, ref_out, atol=2e-5, rtol=1e-4), (out - ref_out).abs().max().item()
    assert torch.allclose(s, ref_s, atol=2e-5, rtol=1e-4), (s - ref_s).abs().max().item()

    w_out, w_s = torch.randn_like(out), torch.randn_like(s)
    ref_grads = torch.autograd.grad((ref_out * w_out).sum() + (ref_s * w_s).sum(), inputs)
    grads = torch.autograd.grad((out * w_out).sum() + (s * w_s).sum(), inputs)
    for name, a, b in zip(["k", "q", "v", "decay", "erase", "write", "s0"], grads, ref_grads):
        assert torch.allclose(a, b, atol=5e-5, rtol=1e-3), (name, (a - b).abs().max().item())


def test_fast_memory_chunked_worst_case_conditioning():
    # Identical keys with maximal erase make (I + L) as non-normal as the model allows.
    k, q, v, decay, erase, write, s0 = random_inputs(b=1, t=256, h=2, seed=1)
    k = k[:, :1].expand_as(k).contiguous()
    erase = torch.full_like(erase, 0.999)
    decay = torch.full_like(decay, 0.9)
    ref_out, ref_s = reference_fast(k, q, v, decay, erase, write, s0)
    out, s = fast_memory_chunked(k, q, v, decay, erase, write, s0, chunk_size=64)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref_out, atol=1e-4, rtol=1e-4), (out - ref_out).abs().max().item()
    assert torch.allclose(s, ref_s, atol=1e-4, rtol=1e-4)


def test_slow_memory_prefix_sum_matches_loop():
    g = torch.Generator().manual_seed(2)
    phi = torch.rand(2, 29, 3, 8, generator=g) + 0.1
    v = torch.randn(2, 29, 3, 8, generator=g)
    num0 = torch.randn(2, 3, 8, generator=g)
    den0 = torch.rand(2, 3, 8, generator=g) + 1.0
    num, den, ref = num0.clone(), den0.clone(), []
    for i in range(phi.size(1)):
        num = num + phi[:, i] * v[:, i]
        den = den + phi[:, i]
        ref.append(num / den.clamp_min(1e-6))
    slow, num_t, den_t = slow_memory_parallel(phi, v, num0, den0, 1e-6)
    assert torch.allclose(slow, torch.stack(ref, dim=1), atol=1e-5, rtol=1e-5)
    assert torch.allclose(num_t, num, atol=1e-5) and torch.allclose(den_t, den, atol=1e-5)


def tiny(**overrides):
    args = dict(
        vocab_size=97, dim=64, n_layers=4, n_heads=4, ffn_hidden=128, max_seq_len=64,
        n_operators=2, operator_rank=4, fast_memory_rank=8, attention_every=4,
        local_kernel=5, dropout=0.0,
    )
    args.update(overrides)
    return IARMXConfig(**args)


def test_model_chunk_scan_matches_loop_scan_forward_and_backward():
    torch.manual_seed(3)
    loop = IARMXForCausalLM(tiny(scan_impl="loop")).train()
    chunk = IARMXForCausalLM(tiny(scan_impl="chunk", scan_chunk_size=4)).train()
    chunk.load_state_dict(loop.state_dict())
    x = torch.randint(0, 97, (2, 23))
    y = torch.randint(0, 97, (2, 23))
    a, b = loop(x, labels=y), chunk(x, labels=y)
    assert torch.allclose(a.logits, b.logits, atol=2e-5, rtol=2e-5), (a.logits - b.logits).abs().max()
    a.loss.backward()
    b.loss.backward()
    for (name, pa), (_, pb) in zip(loop.named_parameters(), chunk.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=2e-5, rtol=1e-3), name


@pytest.mark.parametrize("split", [1, 5])
def test_multi_chunk_scan_cached_decoding_matches_full_sequence(split):
    torch.manual_seed(4)
    m = IARMXForCausalLM(tiny(scan_chunk_size=4)).eval()
    x = torch.randint(0, 97, (1, 19))
    full = m(x).logits
    state, pieces = None, []
    for start in range(0, x.size(1), split):
        out = m(x[:, start : start + split], state=state, use_cache=True)
        state = out.state
        pieces.append(out.logits)
    cached = torch.cat(pieces, dim=1)
    assert torch.allclose(full, cached, atol=2e-5, rtol=2e-5), (full - cached).abs().max().item()


def test_fused_resonance_matches_per_operator_sum():
    torch.manual_seed(5)
    m = ResonanceTransform(dim=64, n_heads=4, n_operators=3, rank=4)
    x, c = torch.randn(2, 7, 4, 16), torch.randn(2, 7, 64)
    y, gates, delta = m(x, c)
    low = torch.einsum("bthd,hodr->bthor", x, m.v)
    resp = torch.einsum("bthor,hodr->bthod", low, m.u)
    ref = (gates.unsqueeze(-1) * resp).sum(-2) / math.sqrt(m.rank)
    assert torch.allclose(delta, ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(y, x + ref, atol=1e-6, rtol=1e-5)
