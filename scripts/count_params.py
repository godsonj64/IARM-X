import argparse
import torch

from iarmx.config import IARMXConfig
from iarmx.model.model import IARMXForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--meta", action="store_true", help="count without allocating real parameter storage")
    args = ap.parse_args()
    cfg = IARMXConfig.from_yaml(args.config)
    if args.meta:
        with torch.device("meta"):
            model = IARMXForCausalLM(cfg)
    else:
        model = IARMXForCausalLM(cfg)
    print(f"parameters={model.num_parameters():,} ({model.num_parameters()/1e9:.3f}B)")
    for i, layer in enumerate(model.layers):
        print(i, layer.__class__.__name__, f"{sum(p.numel() for p in layer.parameters()):,}")


if __name__ == "__main__":
    main()
