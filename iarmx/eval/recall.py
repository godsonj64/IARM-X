import random
import torch


def make_key_value_batch(tokenizer, batch_size=8, pairs=16, distractor_tokens=64, device="cpu"):
    prompts, answers = [], []
    for _ in range(batch_size):
        kv = [(f"K{random.randrange(10**8)}", f"V{random.randrange(10**8)}") for _ in range(pairs)]
        target = random.randrange(pairs)
        body = " ".join(f"{k}={v};" for k, v in kv)
        body += " filler" * distractor_tokens
        prompts.append(body + f" Query: {kv[target][0]}=")
        answers.append(kv[target][1])
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    return enc, answers


@torch.no_grad()
def exact_match_recall(model, tokenizer, **kwargs):
    enc, answers = make_key_value_batch(
        tokenizer, device=next(model.parameters()).device, **kwargs
    )
    mask = enc.get("attention_mask")
    out = model.generate(
        enc.input_ids,
        attention_mask=mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=12,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    preds = []
    for i in range(out.size(0)):
        prompt_len = int(mask[i].sum().item()) if mask is not None else enc.input_ids.size(1)
        preds.append(tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True))
    score = sum(a in p for a, p in zip(answers, preds)) / len(answers)
    return {"kv_recall_exact_match": score, "predictions": preds, "answers": answers}
