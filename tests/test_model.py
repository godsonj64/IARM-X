import tempfile
from pathlib import Path

import torch

from iarmx import IARMXConfig, IARMXForCausalLM
from iarmx.model.state import RecurrentLayerState, AttentionLayerState


def tiny(**overrides):
    args = dict(
        vocab_size=97,
        dim=64,
        n_layers=4,
        n_heads=4,
        ffn_hidden=128,
        max_seq_len=64,
        n_operators=2,
        operator_rank=4,
        fast_memory_rank=8,
        attention_every=4,
        local_kernel=5,
        dropout=0.0,
    )
    args.update(overrides)
    return IARMXConfig(**args)


def test_forward_shape_and_loss():
    torch.manual_seed(0)
    cfg = tiny()
    m = IARMXForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))
    out = m(x, labels=y)
    assert out.logits.shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(out.loss)


def test_full_vs_token_cached_logits_with_conv_and_importance():
    torch.manual_seed(1)
    cfg = tiny()
    m = IARMXForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    full = m(x).logits
    state = None
    pieces = []
    for i in range(x.size(1)):
        out = m(x[:, i : i + 1], state=state, use_cache=True)
        state = out.state
        pieces.append(out.logits)
    cached = torch.cat(pieces, dim=1)
    assert torch.allclose(full, cached, atol=2e-5, rtol=2e-5), (full - cached).abs().max().item()


def test_chunked_vs_full_logits():
    torch.manual_seed(2)
    cfg = tiny()
    m = IARMXForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 13))
    full = m(x).logits
    first = m(x[:, :5], use_cache=True)
    second = m(x[:, 5:], state=first.state, use_cache=True)
    chunked = torch.cat([first.logits, second.logits], dim=1)
    assert torch.allclose(full, chunked, atol=2e-5, rtol=2e-5), (full - chunked).abs().max().item()


def test_no_future_leakage():
    torch.manual_seed(3)
    cfg = tiny()
    m = IARMXForCausalLM(cfg).eval()
    prefix = torch.randint(0, cfg.vocab_size, (1, 7))
    a = torch.cat([prefix, torch.randint(0, cfg.vocab_size, (1, 5))], dim=1)
    b = torch.cat([prefix, torch.randint(0, cfg.vocab_size, (1, 5))], dim=1)
    la = m(a).logits[:, : prefix.size(1)]
    lb = m(b).logits[:, : prefix.size(1)]
    assert torch.equal(la, lb)


def test_tied_safetensors_roundtrip():
    torch.manual_seed(4)
    cfg = tiny(tie_embeddings=True)
    m = IARMXForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 9))
    ref = m(x).logits
    with tempfile.TemporaryDirectory() as td:
        m.save_pretrained(td)
        loaded = IARMXForCausalLM.from_pretrained(td).eval()
        got = loaded(x).logits
        assert loaded.embed.weight.data_ptr() == loaded.lm_head.weight.data_ptr()
        assert torch.equal(ref, got)
        assert (Path(td) / "model.safetensors").exists()


def _state_numel(state, recurrent_only=False, attention_only=False):
    total = 0
    for st in state.layers:
        if isinstance(st, RecurrentLayerState) and not attention_only:
            total += st.slow_num.numel() + st.slow_den.numel() + st.fast.numel() + st.conv_cache.numel()
        if isinstance(st, AttentionLayerState) and not recurrent_only:
            total += st.key.numel() + st.value.numel()
            if st.importance is not None:
                total += st.importance.numel()
    return total


def test_recurrent_state_constant_attention_cache_grows():
    torch.manual_seed(5)
    cfg = tiny()
    m = IARMXForCausalLM(cfg).eval()
    s1 = m(torch.randint(0, cfg.vocab_size, (1, 1)), use_cache=True).state
    s16 = m(torch.randint(0, cfg.vocab_size, (1, 16)), use_cache=True).state
    assert _state_numel(s1, recurrent_only=True) == _state_numel(s16, recurrent_only=True)
    assert _state_numel(s16, attention_only=True) > _state_numel(s1, attention_only=True)


def test_fp32_persistent_state_under_bfloat16():
    torch.manual_seed(6)
    cfg = tiny(n_layers=1, attention_every=0, max_seq_len=512)
    m = IARMXForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    # q=0 -> phi=ELU(0)+1=1 exactly, so den must equal sequence length.
    m.layers[0].q_proj.weight.data.zero_()
    x = torch.randint(0, cfg.vocab_size, (1, 300))
    out = m(x, use_cache=True)
    st = out.state.layers[0]
    assert st.slow_num.dtype == torch.float32
    assert st.slow_den.dtype == torch.float32
    assert st.fast.dtype == torch.float32
    assert torch.all(st.slow_den == 300.0)


def test_gradient_checkpointing_matches_forward_and_backward():
    torch.manual_seed(7)
    cfg_a = tiny(gradient_checkpointing=False)
    cfg_b = tiny(gradient_checkpointing=True)
    a = IARMXForCausalLM(cfg_a).train()
    b = IARMXForCausalLM(cfg_b).train()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, cfg_a.vocab_size, (2, 10))
    y = torch.randint(0, cfg_a.vocab_size, (2, 10))
    oa = a(x, labels=y)
    ob = b(x, labels=y)
    assert torch.allclose(oa.logits, ob.logits, atol=1e-6, rtol=1e-6)
    oa.loss.backward()
    ob.loss.backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert pa.grad is not None and pb.grad is not None
        assert torch.allclose(pa.grad, pb.grad, atol=2e-5, rtol=2e-4), na


def test_all_importance_predictors_receive_gradient():
    torch.manual_seed(8)
    cfg = tiny()
    m = IARMXForCausalLM(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    y = torch.randint(0, cfg.vocab_size, (2, 10))
    m(x, labels=y).loss.backward()
    names = []
    for name, p in m.named_parameters():
        if ".importance." in name or name.endswith("importance_scale"):
            names.append(name)
            assert p.grad is not None, name
            assert torch.count_nonzero(p.grad).item() > 0, name
    assert len(names) >= 7  # 3 recurrent weight+bias pairs + attention scale


def test_padded_batch_generation_matches_individual_generation():
    torch.manual_seed(9)
    cfg = tiny()
    m = IARMXForCausalLM(cfg).eval()
    pad = 0
    p1 = torch.tensor([11, 12, 13, 14])
    p2 = torch.tensor([21, 22])
    padded = torch.tensor([[11, 12, 13, 14], [21, 22, pad, pad]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    batched = m.generate(
        padded,
        attention_mask=mask,
        pad_token_id=pad,
        max_new_tokens=3,
        temperature=0.0,
    )
    a = m.generate(p1[None], max_new_tokens=3, temperature=0.0).squeeze(0)
    b = m.generate(p2[None], max_new_tokens=3, temperature=0.0).squeeze(0)
    assert torch.equal(batched[0, : a.numel()], a)
    assert torch.equal(batched[1, : b.numel()], b)
