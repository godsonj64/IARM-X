import argparse
from pathlib import Path
import time
import yaml
import torch

from iarmx import IARMXConfig, IARMXForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq", type=int, default=512)
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    cfg = IARMXConfig.from_dict(spec.get("model", spec))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IARMXForCausalLM(cfg).to(device).eval()
    x = torch.randint(0, cfg.vocab_size, (1, args.seq), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print({"seconds": dt, "tokens_per_second": args.seq / dt, "state_position": out.state.position})


if __name__ == "__main__":
    main()
