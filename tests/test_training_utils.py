import tempfile
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset

from iarmx import IARMXConfig, IARMXForCausalLM
from iarmx.training.checkpoint import save_checkpoint, load_checkpoint
from iarmx.training.train import shard_dataset


def small_model():
    return IARMXForCausalLM(
        IARMXConfig(
            vocab_size=31,
            dim=32,
            n_layers=2,
            n_heads=4,
            ffn_hidden=64,
            max_seq_len=16,
            n_operators=2,
            operator_rank=4,
            fast_memory_rank=4,
            attention_every=2,
        )
    )


def test_checkpoint_restores_model_optimizer_scheduler_and_step():
    torch.manual_seed(0)
    m = small_model()
    opt = AdamW(m.parameters(), lr=1e-3)
    sched = LambdaLR(opt, lambda _: 1.0)
    x = torch.randint(0, 31, (1, 5))
    y = torch.randint(0, 31, (1, 5))
    m(x, labels=y).loss.backward()
    opt.step()
    sched.step()
    ref = {k: v.detach().clone() for k, v in m.state_dict().items()}

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "step.pt"
        save_checkpoint(p, m, opt, sched, step=7)
        m2 = small_model()
        opt2 = AdamW(m2.parameters(), lr=1e-3)
        sched2 = LambdaLR(opt2, lambda _: 1.0)
        ckpt = load_checkpoint(p, m2, opt2, sched2)
        assert ckpt["step"] == 7
        for k, v in m2.state_dict().items():
            assert torch.equal(v, ref[k]), k
        assert sched2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]


def test_distributed_sampler_partitions_map_dataset():
    ds = TensorDataset(torch.arange(20))
    _, s0 = shard_dataset(ds, world=2, rank=0, streaming=False, seed=42)
    _, s1 = shard_dataset(ds, world=2, rank=1, streaming=False, seed=42)
    a, b = set(iter(s0)), set(iter(s1))
    assert a.isdisjoint(b)
    assert a | b == set(range(20))
