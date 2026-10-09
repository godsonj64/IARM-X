"""Analytical training-compute and dollar-cost model for an IARM-X config.

Counts multiply-accumulates (MACs) per token for every matmul in the model,
including the chunkwise fast-memory scan and causal attention, then converts
the total into hours, dollars and kWh for a set of accelerators.

    python scripts/estimate_cost.py --config configs/iarmx_100m_pretrain.yaml
    python scripts/estimate_cost.py --config configs/iarmx_1.3b.yaml --mfu 0.3 0.45

Prices are representative marketplace rates from public Sept-Oct 2026
snapshots and move daily: pass --price NAME=USD_PER_HOUR to override.
"""

import argparse
from pathlib import Path

import yaml

from iarmx.config import IARMXConfig

# name: (dense BF16 tensor TFLOP/s with FP32 accumulate, USD/hour, board watts)
HARDWARE = {
    "rtx3090": (71.0, 0.15, 350),
    "rtx4090": (165.2, 0.30, 450),
    "rtx5090": (209.5, 0.35, 575),
    "h100-pcie-spot": (756.0, 0.90, 350),
    "h100-sxm": (989.0, 1.74, 700),
}


def macs_per_token(cfg: IARMXConfig, seq_len: int) -> dict[str, float]:
    d, h, dh, f, v = cfg.dim, cfg.n_heads, cfg.head_dim, cfg.ffn_hidden, cfg.vocab_size
    o, rk, r = cfg.n_operators, cfg.operator_rank, cfg.fast_memory_rank
    c = min(cfg.scan_chunk_size, seq_len)
    n_attn = sum(cfg.is_attention_layer(i) for i in range(cfg.n_layers))
    n_rec = cfg.n_layers - n_attn

    resonance = 2 * (d * h * o + 2 * d * o * rk)  # q and k transforms
    # Chunkwise scan per token and head: K K^T, Q K^T, triangular solve,
    # intra-chunk read, and the three state products (W~ S, Q S, K^T Delta).
    scan = h * (2 * c * r + c * (dh + r) / 2 + c * dh + 3 * r * dh)
    return {
        "qkv/out projections": n_rec * 5 * d * d + n_attn * 4 * d * d,  # rec adds out_gate
        "resonance operators": cfg.n_layers * resonance,
        "SwiGLU FFN": cfg.n_layers * 3 * d * f,
        "memory/gate heads": n_rec * (2 * d * r + 7 * d * h + d * cfg.local_kernel),
        "recurrent scan": n_rec * scan,
        "exact attention": n_attn * d * seq_len,  # causal QK^T + AV, ~T/2 keys each
        "LM head": d * v,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", type=float, help="training tokens (default: config target_tokens or 1e10)")
    ap.add_argument("--seq-len", type=int, help="training context (default: config data.seq_len)")
    ap.add_argument("--budget", type=float, default=50.0, help="USD budget for the throughput target")
    ap.add_argument("--mfu", type=float, nargs="+", default=[0.25, 0.40])
    ap.add_argument("--checkpointing", choices=["config", "on", "off"], default="config")
    ap.add_argument("--price", action="append", default=[], metavar="NAME=USD_PER_HOUR")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    cfg = IARMXConfig.from_dict(spec.get("model", spec))
    train = spec.get("training", {})
    tokens = args.tokens or float(train.get("target_tokens") or 1e10)
    seq_len = args.seq_len or int(spec.get("data", {}).get("seq_len", cfg.max_seq_len))
    ckpt = cfg.gradient_checkpointing if args.checkpointing == "config" else args.checkpointing == "on"
    hardware = dict(HARDWARE)
    for item in args.price:
        name, price = item.split("=")
        peak, _, watts = hardware[name]
        hardware[name] = (peak, float(price), watts)

    parts = macs_per_token(cfg, seq_len)
    total = sum(parts.values())
    layer_macs = total - parts["LM head"]
    # forward = 2 FLOPs/MAC, backward = 2x forward; checkpointing re-runs layer forwards.
    flops_tok = 6 * total + (2 * layer_macs if ckpt else 0)
    flops = flops_tok * tokens

    print(f"config={args.config} seq_len={seq_len} tokens={tokens:.3g} checkpointing={ckpt}")
    print(f"forward MACs/token = {total / 1e6:.1f}M")
    for name, value in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {value / 1e6:8.2f}M  {100 * value / total:5.1f}%")
    print(f"training FLOPs/token = {flops_tok / 1e9:.3f} GFLOP; total = {flops:.3e} FLOP")
    print()
    print(f"{'hardware':16s} {'MFU':>4s} {'tok/s':>9s} {'hours':>8s} {'USD':>8s} {'kWh':>7s}"
          f" {'tok/s for $' + format(args.budget, 'g'):>14s}")
    for name, (peak, price, watts) in hardware.items():
        need = tokens * price / (3600 * args.budget)
        for mfu in args.mfu:
            rate = peak * 1e12 * mfu / flops_tok
            hours = tokens / rate / 3600
            print(f"{name:16s} {mfu:4.2f} {rate:9,.0f} {hours:8.1f} {hours * price:8.2f}"
                  f" {hours * watts / 1000:7.1f} {need:14,.0f}")


if __name__ == "__main__":
    main()
