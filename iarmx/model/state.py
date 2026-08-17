from dataclasses import dataclass
import torch


@dataclass
class RecurrentLayerState:
    slow_num: torch.Tensor
    slow_den: torch.Tensor
    fast: torch.Tensor
    conv_cache: torch.Tensor


@dataclass
class AttentionLayerState:
    key: torch.Tensor
    value: torch.Tensor
    importance: torch.Tensor | None = None


@dataclass
class IARMXState:
    layers: list[RecurrentLayerState | AttentionLayerState | None]
    position: int = 0
