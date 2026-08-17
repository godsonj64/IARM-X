import argparse
import torch
from .model.model import IARMXForCausalLM
from .data.tokenizer import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = IARMXForCausalLM.from_pretrained(args.model, device=device, dtype=dtype)
    tok = load_tokenizer(args.tokenizer)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, eos_token_id=tok.eos_token_id)
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__": main()
