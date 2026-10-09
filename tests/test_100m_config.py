from pathlib import Path

import torch
import yaml

from iarmx import IARMXConfig, IARMXForCausalLM
from iarmx.training.train import estimate_total_steps, should_stop


def test_100m_config_parameter_count():
    spec = yaml.safe_load(Path("configs/iarmx_100m_pretrain.yaml").read_text())
    cfg = IARMXConfig.from_dict(spec["model"])
    with torch.device("meta"):
        model = IARMXForCausalLM(cfg)
    assert model.num_parameters() == 100_202_256


def test_token_budget_scheduler_horizon_scales_with_world_size():
    train = {"micro_batch_size": 8, "grad_accum": 16, "target_tokens": 10_000_000_000}
    data = {"seq_len": 512}
    one = estimate_total_steps(train, data, world=1)
    two = estimate_total_steps(train, data, world=2)
    assert one == 152_588
    assert two == 76_294


def test_budget_stop_conditions():
    assert not should_stop(10, 99, 0, {"target_tokens": 100})
    assert should_stop(10, 100, 0, {"target_tokens": 100})
    assert not should_stop(3, 0, 19, {"epochs": 2}, dataset_len=10)
    assert should_stop(3, 0, 20, {"epochs": 2}, dataset_len=10)


def test_100m_configs_default_to_2048_context_with_unchanged_tokens_per_step():
    root = Path(__file__).resolve().parents[1]
    for name in ["iarmx_100m_pretrain", "iarmx_100m_pretrain_local", "iarmx_100m_pretrain_memmap"]:
        spec = yaml.safe_load((root / "configs" / f"{name}.yaml").read_text())
        train, data = spec["training"], spec["data"]
        assert data["seq_len"] == 2048 <= spec["model"]["max_seq_len"], name
        assert train["micro_batch_size"] * train["grad_accum"] * data["seq_len"] == 65_536, name
        assert estimate_total_steps(train, data, world=1) == 152_588, name
    for name in ["iarmx_100m_sft", "iarmx_100m_sft_local"]:
        spec = yaml.safe_load((root / "configs" / f"{name}.yaml").read_text())
        assert spec["data"]["seq_len"] == 2048 <= spec["model"]["max_seq_len"], name
