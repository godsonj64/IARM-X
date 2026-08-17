import torch

from iarmx import IARMXConfig, IARMXForCausalLM


def main():
    cfg = IARMXConfig(
        vocab_size=257,
        dim=128,
        n_layers=4,
        n_heads=4,
        ffn_hidden=256,
        max_seq_len=64,
        n_operators=4,
        operator_rank=8,
        fast_memory_rank=16,
        attention_every=4,
    )
    model = IARMXForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, labels=y, use_cache=True)
    print(
        "loss",
        float(out.loss.detach()),
        "logits",
        tuple(out.logits.shape),
        "params",
        model.num_parameters(),
    )


if __name__ == "__main__":
    main()
