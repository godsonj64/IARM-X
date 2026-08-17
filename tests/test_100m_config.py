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
