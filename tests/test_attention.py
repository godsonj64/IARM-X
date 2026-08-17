from unittest.mock import patch

import torch

from iarmx import IARMXConfig
from iarmx.model.attention import ResonanceAttentionBlock


def cfg():
    return IARMXConfig(
        vocab_size=100,
        dim=64,
        n_layers=1,
        n_heads=4,
        ffn_hidden=128,
        max_seq_len=32,
        n_operators=2,
        operator_rank=4,
        fast_memory_rank=8,
        attention_every=1,
        use_sdpa=True,
        dropout=0.0,
    )


def test_sdpa_is_used_even_with_importance_bias():
    torch.manual_seed(0)
    c = cfg()
    block = ResonanceAttentionBlock(c).eval()
    x = torch.randn(2, 6, c.dim)
    pos = torch.arange(6)
    imp = torch.rand(2, 6, c.n_heads).clamp_min(0.1)
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=wrapped):
        _, state, _ = block(x, pos, historical_importance=imp)
    assert calls
    assert state.importance is not None
    assert state.importance.shape == (2, c.n_heads, 6)


def test_importance_cache_extends_during_decode():
    torch.manual_seed(1)
    c = cfg()
    block = ResonanceAttentionBlock(c).eval()
    x = torch.randn(1, 5, c.dim)
    imp = torch.rand(1, 5, c.n_heads).clamp_min(0.1)
    _, state, _ = block(x, torch.arange(5), historical_importance=imp)
    x2 = torch.randn(1, 1, c.dim)
    imp2 = torch.rand(1, 1, c.n_heads).clamp_min(0.1)
    _, state2, _ = block(x2, torch.tensor([5]), state=state, historical_importance=imp2)
    assert state2.importance.shape[-1] == 6
    assert torch.allclose(state2.importance[..., :5], state.importance)


def test_sdpa_importance_bias_matches_manual_attention():
    torch.manual_seed(2)
    c_sdpa = cfg()
    c_manual = IARMXConfig.from_dict({**c_sdpa.to_dict(), "use_sdpa": False})
    a = ResonanceAttentionBlock(c_sdpa).eval()
    b = ResonanceAttentionBlock(c_manual).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 7, c_sdpa.dim)
    imp = torch.rand(1, 7, c_sdpa.n_heads).clamp_min(0.1)
    ya, sa, _ = a(x, torch.arange(7), historical_importance=imp)
    yb, sb, _ = b(x, torch.arange(7), historical_importance=imp)
    assert torch.allclose(ya, yb, atol=2e-6, rtol=2e-6), (ya - yb).abs().max().item()

    x2 = torch.randn(1, 1, c_sdpa.dim)
    imp2 = torch.rand(1, 1, c_sdpa.n_heads).clamp_min(0.1)
    ya2, _, _ = a(x2, torch.tensor([7]), state=sa, historical_importance=imp2)
    yb2, _, _ = b(x2, torch.tensor([7]), state=sb, historical_importance=imp2)
    assert torch.allclose(ya2, yb2, atol=2e-6, rtol=2e-6), (ya2 - yb2).abs().max().item()
