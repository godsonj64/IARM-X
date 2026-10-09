import pytest
import torch

from iarmx.generate import chat_prompt_ids
from iarmx.training.train import apply_overrides
from iarmx.utils.device import inference_dtype, resolve_precision


def test_overrides_parse_yaml_values_and_create_sections():
    spec = {"training": {"epochs": 2, "lr": 0.1}, "data": {"seq_len": 512}}
    apply_overrides(spec, [
        "training.target_tokens=1e8",
        "training.epochs=null",
        "data.seq_len=2048",
        "model.gradient_checkpointing=false",
        "training.output_dir=/content/drive/MyDrive/runs",
    ])
    assert spec["training"]["target_tokens"] == 100_000_000
    assert spec["training"]["epochs"] is None
    assert spec["data"]["seq_len"] == 2048
    assert spec["model"]["gradient_checkpointing"] is False
    assert spec["training"]["output_dir"] == "/content/drive/MyDrive/runs"
    with pytest.raises(ValueError):
        apply_overrides(spec, ["training.lr"])


@pytest.mark.parametrize(
    "requested,capability,expected",
    [
        ("bf16", (8, 0), "bf16"),  # A100/H100/RTX 30-50xx
        ("bf16", (7, 5), "fp16"),  # T4: emulated bf16 would be very slow
        ("auto", (7, 0), "fp16"),  # V100
        ("auto", (9, 0), "bf16"),
        ("fp16", (9, 0), "fp16"),
        ("fp32", (7, 5), "fp32"),
    ],
)
def test_precision_falls_back_to_fp16_without_native_bf16(monkeypatch, requested, capability, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    assert resolve_precision(requested, "cuda") == expected


def test_cpu_trains_in_fp32_and_rejects_unknown_precision():
    assert resolve_precision("bf16", "cpu") == "fp32"
    assert inference_dtype("cpu") == torch.float32
    with pytest.raises(ValueError):
        resolve_precision("int8", "cpu")


class ChatTokenizer:
    ids = {"<|user|>": 1, "<|assistant|>": 2, "<|end|>": 3}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def encode(self, text, add_special_tokens=False, split_special_tokens=False):
        assert split_special_tokens
        return [10 + i for i, _ in enumerate(text.split())]


def test_chat_prompt_matches_the_sft_turn_format():
    ids, stop = chat_prompt_ids(ChatTokenizer(), "hello there")
    # data/sft.py renders a user turn as <|user|> text <|end|>, then <|assistant|> reply <|end|>.
    assert ids == [1, 10, 11, 3, 2]
    assert stop == 3
