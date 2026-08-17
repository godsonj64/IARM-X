from pathlib import Path

import torch
import yaml

from iarmx import IARMXConfig, IARMXForCausalLM


def test_research_config_is_actually_about_1_3b_parameters():
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "configs" / "iarmx_1.3b.yaml").read_text())
    cfg = IARMXConfig.from_dict(spec["model"])
    with torch.device("meta"):
        model = IARMXForCausalLM(cfg)
    n = model.num_parameters()
    assert n == 1_309_384_768
    assert 1.28e9 <= n <= 1.32e9
