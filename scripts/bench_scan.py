"""Compare the per-token loop scan with the chunkwise scan on one recurrent block.

Reports forward+backward wall time, matmul FLOPs (FlopCounterMode) and the
number of non-view ATen ops dispatched. On a GPU each non-view op is roughly
one kernel launch, so the op count is what bounds the loop's throughput there.

    python scripts/bench_scan.py --config configs/iarmx_100m_pretrain.yaml --seq 512
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
import yaml
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

from iarmx.config import IARMXConfig
from iarmx.model.recurrent import IARMXRecurrentBlock


class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not getattr(func, "is_view", False) and func.overloadpacket is not torch.ops.aten.detach:
            self.ops += 1
        return func(*args, **(kwargs or {}))


def step(block, x, positions):
    y, _, aux = block(x, positions)
    (y.float().square().mean() + aux["importance"].float().mean()).backward()
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/iarmx_100m_pretrain.yaml")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    model_cfg = dict(spec.get("model", spec), dropout=0.0, gradient_checkpointing=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = args.batch * args.seq
    torch.manual_seed(0)
    x = torch.randn(args.batch, args.seq, model_cfg["dim"], device=device)
    positions = torch.arange(args.seq, device=device)

    results = {}
    reference = None
    for impl in ("loop", "chunk"):
        cfg = IARMXConfig.from_dict(dict(model_cfg, scan_impl=impl, scan_chunk_size=args.chunk))
        torch.manual_seed(1)
        block = IARMXRecurrentBlock(cfg).to(device)
        counter, flops = OpCounter(), FlopCounterMode(display=False)
        with counter, flops:
            y = step(block, x, positions)
        if reference is None:
            reference = y.detach()
        # FlopCounterMode charges depthwise convolution_backward as a dense d x d conv.
        matmul_flops = sum(
            f for op, f in flops.get_flop_counts()["Global"].items()
            if op is not torch.ops.aten.convolution_backward
        )
        diff = (y.detach() - reference).abs().max().item()
        times = []
        for _ in range(args.reps):
            block.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            step(block, x, positions)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        sec = statistics.median(times)
        results[impl] = sec
        print(
            f"{impl:5s} fwd+bwd {sec * 1e3:9.1f} ms  {tokens / sec:10,.0f} tok/s/block  "
            f"ops={counter.ops:7,d} ({counter.ops / tokens:6.1f}/token)  "
            f"matmul GFLOP={matmul_flops / 1e9:7.2f}  max|dy| vs loop={diff:.2e}"
        )
    print(f"device={device} threads={torch.get_num_threads()} speedup={results['loop'] / results['chunk']:.1f}x")


if __name__ == "__main__":
    main()
