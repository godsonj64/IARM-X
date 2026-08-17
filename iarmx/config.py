from dataclasses import dataclass, asdict
from pathlib import Path
import json
import yaml


@dataclass
class IARMXConfig:
    vocab_size: int = 50260
    dim: int = 768
    n_layers: int = 11
    n_heads: int = 12
    ffn_hidden: int = 2048
    max_seq_len: int = 4096
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6

    # IARM resonance (source-derived core)
    n_operators: int = 4
    operator_rank: int = 16
    local_kernel: int = 5

    # IARM-X extensions
    attention_every: int = 4
    fast_memory_rank: int = 32
    slow_memory_eps: float = 1e-6
    memory_decay_min: float = 0.90
    memory_decay_max: float = 0.9999
    attention_bias_from_importance: bool = True
    importance_scale_init: float = 0.10
    tie_embeddings: bool = True

    # implementation
    gradient_checkpointing: bool = False
    use_sdpa: bool = True

    def __post_init__(self):
        if self.dim <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("dim, n_layers and n_heads must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.n_operators <= 0 or self.operator_rank <= 0:
            raise ValueError("n_operators and operator_rank must be positive")
        if self.local_kernel <= 0:
            raise ValueError("local_kernel must be positive")
        if self.fast_memory_rank <= 0:
            raise ValueError("fast_memory_rank must be positive")
        if not (0.0 <= self.memory_decay_min <= self.memory_decay_max <= 1.0):
            raise ValueError("memory decay bounds must satisfy 0 <= min <= max <= 1")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def is_attention_layer(self, layer_idx: int) -> bool:
        if self.attention_every <= 0:
            return False
        return (layer_idx + 1) % self.attention_every == 0

    def to_dict(self):
        return asdict(self)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path):
        data = yaml.safe_load(Path(path).read_text())
        model = data.get("model", data)
        return cls.from_dict(model)
