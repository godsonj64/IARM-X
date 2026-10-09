import argparse
import torch
from .model.model import IARMXForCausalLM
from .data.tokenizer import load_tokenizer
from .utils.device import inference_dtype


def chat_prompt_ids(tokenizer, prompt: str):
    """Prompt ids in the UltraChat SFT format (see data/sft.py) and the id that ends a reply."""
    user, assistant, end = (
        tokenizer.convert_tokens_to_ids(t) for t in ("<|user|>", "<|assistant|>", "<|end|>")
    )
    body = tokenizer.encode(prompt, add_special_tokens=False, split_special_tokens=True)
    return [user] + body + [end, assistant], end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--chat", action="store_true", help="wrap the prompt as a user turn (SFT models)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding")
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IARMXForCausalLM.from_pretrained(args.model, device=device, dtype=inference_dtype(device))
    tok = load_tokenizer(args.tokenizer)
    if args.chat:
        prompt_ids, stop_id = chat_prompt_ids(tok, args.prompt)
    else:
        prompt_ids, stop_id = tok(args.prompt).input_ids, tok.eos_token_id
    ids = torch.tensor([prompt_ids], device=device)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=stop_id,
    )
    new = out[0, ids.size(1):] if args.chat else out[0]
    print(tok.decode(new, skip_special_tokens=True).strip())

if __name__ == "__main__": main()
