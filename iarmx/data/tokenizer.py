def load_tokenizer(name: str = "gpt2"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    specials = {"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end|>"]}
    tok.add_special_tokens(specials)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok
