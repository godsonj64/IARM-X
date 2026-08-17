from dataclasses import dataclass
from pathlib import Path
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file, load_file

from ..config import IARMXConfig
from .layers import RMSNorm
from .recurrent import IARMXRecurrentBlock
from .attention import ResonanceAttentionBlock
from .state import IARMXState


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    state: IARMXState | None = None
    aux: list[dict] | None = None


class IARMXForCausalLM(nn.Module):
    def __init__(self, cfg: IARMXConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([
            ResonanceAttentionBlock(cfg) if cfg.is_attention_layer(i) else IARMXRecurrentBlock(cfg)
            for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.dim, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def initial_state(self, batch: int, device=None, dtype=None) -> IARMXState:
        device = device or self.embed.weight.device
        dtype = dtype or self.embed.weight.dtype
        states = [layer.initial_state(batch, device, dtype) for layer in self.layers]
        return IARMXState(states, position=0)

    def _run_recurrent_checkpointed(self, layer, x, positions):
        def fn(inp):
            y, _, aux = layer(inp, positions, None)
            return y, aux["importance"]

        return checkpoint(fn, x, use_reentrant=False)

    def _run_attention_checkpointed(self, layer, x, positions, historical_importance):
        def fn(inp):
            y, _, _ = layer(inp, positions, None, historical_importance=historical_importance)
            return y

        return checkpoint(fn, x, use_reentrant=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        state: IARMXState | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        b, t = input_ids.shape
        if t == 0:
            raise ValueError("input_ids must contain at least one token")
        if t > self.cfg.max_seq_len and state is None:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}")

        x = self.embed(input_ids)
        start = state.position if state is not None else 0
        positions = torch.arange(start, start + t, device=input_ids.device)
        new_states, aux_all = [], []

        # Importance from every recurrent layer in a group contributes to the next
        # attention layer. This prevents early recurrent importance predictors from
        # becoming permanently dead parameters in an R,R,R,A schedule.
        importance_sum = None
        importance_count = 0
        checkpointing = self.cfg.gradient_checkpointing and self.training and state is None and not use_cache

        for i, layer in enumerate(self.layers):
            st = state.layers[i] if state is not None else None
            if isinstance(layer, IARMXRecurrentBlock):
                if checkpointing:
                    x, importance = self._run_recurrent_checkpointed(layer, x, positions)
                    new_st = None
                    aux = {"importance": importance}
                else:
                    x, new_st, aux = layer(x, positions, st)
                    importance = aux["importance"]
                importance_sum = importance if importance_sum is None else importance_sum + importance
                importance_count += 1
            else:
                historical_importance = (
                    importance_sum / importance_count if importance_count > 0 else None
                )
                if checkpointing:
                    x = self._run_attention_checkpointed(
                        layer, x, positions, historical_importance
                    )
                    new_st = None
                    aux = {}
                else:
                    x, new_st, aux = layer(
                        x, positions, st, historical_importance=historical_importance
                    )
                importance_sum = None
                importance_count = 0
            new_states.append(new_st)
            aux_all.append(aux)

        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )
        out_state = IARMXState(new_states, start + t) if use_cache else None
        return CausalLMOutput(logits=logits, loss=loss, state=out_state, aux=aux_all)

    @torch.no_grad()
    def _generate_dense(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        state = None
        out_ids = input_ids
        out = self(input_ids, state=None, use_cache=True)
        state = out.state
        logits = out.logits[:, -1]
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cdf = sorted_probs.cumsum(-1)
                mask = cdf - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(mask, 0)
                sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp_min(1e-12)
                choice = torch.multinomial(sorted_probs, 1)
                nxt = sorted_idx.gather(-1, choice)

            if eos_token_id is not None:
                eos_fill = torch.full_like(nxt, eos_token_id)
                nxt = torch.where(finished[:, None], eos_fill, nxt)
            out_ids = torch.cat([out_ids, nxt], dim=1)

            if eos_token_id is not None:
                finished = finished | (nxt.squeeze(-1) == eos_token_id)
                if torch.all(finished):
                    break

            out = self(nxt, state=state, use_cache=True)
            state, logits = out.state, out.logits[:, -1]
        return out_ids

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        eos_token_id: int | None = None,
        attention_mask: torch.Tensor | None = None,
        pad_token_id: int | None = None,
    ):
        """Autoregressive generation with exact recurrent/attention caching.

        If a padded batch is supplied, each prompt is compacted using
        ``attention_mask`` and decoded independently. This is slower than a
        length-bucketed production scheduler but is mathematically correct for
        variable-length recurrent states and avoids treating padding as history.
        """
        self.eval()
        if attention_mask is None or bool(torch.all(attention_mask == 1)):
            return self._generate_dense(
                input_ids, max_new_tokens, temperature, top_p, eos_token_id
            )

        outputs = []
        for row in range(input_ids.size(0)):
            valid = attention_mask[row].to(torch.bool)
            prompt = input_ids[row][valid].unsqueeze(0)
            if prompt.numel() == 0:
                raise ValueError("each generation prompt must contain at least one non-padding token")
            outputs.append(
                self._generate_dense(
                    prompt, max_new_tokens, temperature, top_p, eos_token_id
                ).squeeze(0)
            )
        max_len = max(o.numel() for o in outputs)
        fill_id = pad_token_id
        if fill_id is None:
            fill_id = eos_token_id if eos_token_id is not None else 0
        result = torch.full(
            (len(outputs), max_len), fill_id, device=input_ids.device, dtype=input_ids.dtype
        )
        for i, seq in enumerate(outputs):
            result[i, : seq.numel()] = seq
        return result

    def save_pretrained(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.cfg.save(path / "config.json")
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        # Safetensors rejects shared storage. With tied embeddings the two names
        # refer to one Parameter, so persist a single copy and restore the tie on load.
        if self.cfg.tie_embeddings:
            state.pop("lm_head.weight", None)
        save_file(state, str(path / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, path: str | Path, device="cpu", dtype=None):
        path = Path(path)
        cfg = IARMXConfig.from_dict(json.loads((path / "config.json").read_text()))
        model = cls(cfg)
        sd = load_file(str(path / "model.safetensors"), device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        allowed_missing = {"lm_head.weight"} if cfg.tie_embeddings else set()
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        if dtype is not None:
            model = model.to(dtype=dtype)
        return model.to(device)
