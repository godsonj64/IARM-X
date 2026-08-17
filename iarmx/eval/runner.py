import argparse
import json
import torch
from ..model.model import IARMXForCausalLM
from ..data.tokenizer import load_tokenizer
from .recall import exact_match_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--pairs", type=int, default=16)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = IARMXForCausalLM.from_pretrained(args.model, device=device, dtype=dtype)
    tok = load_tokenizer(args.tokenizer)
    result = exact_match_recall(model, tok, pairs=args.pairs)
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
