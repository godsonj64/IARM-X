def render_ultrachat(messages, tokenizer):
    """Render assistant-only next-token supervision without target leakage."""
    user_id = tokenizer.convert_tokens_to_ids("<|user|>")
    asst_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")

    full_ids = []
    supervise = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        toks = tokenizer.encode(content, add_special_tokens=False)
        is_assistant = role == "assistant"
        prefix = asst_id if is_assistant else user_id
        seg = [prefix] + toks + [end_id]
        full_ids.extend(seg)
        supervise.extend([False] + [is_assistant] * len(toks) + [is_assistant])

    if len(full_ids) < 2:
        return [], []
    input_ids = full_ids[:-1]
    labels = [
        full_ids[i + 1] if supervise[i + 1] else -100
        for i in range(len(full_ids) - 1)
    ]
    return input_ids, labels


def build_sft_dataset(
    tokenizer,
    name="HuggingFaceH4/ultrachat_200k",
    split="train_sft",
    seq_len=512,
    data_files=None,
):
    from datasets import load_dataset

    if data_files:
        ds = load_dataset("parquet", data_files=data_files, split=split)
    else:
        ds = load_dataset(name, split=split)

    def fn(ex):
        ids, labels = render_ultrachat(ex["messages"], tokenizer)
        return {"input_ids": ids[:seq_len], "labels": labels[:seq_len]}

    return ds.map(fn, remove_columns=ds.column_names)


def collate_sft(batch, pad_id: int):
    import torch

    n = max(len(x["input_ids"]) for x in batch)
    xs, ys = [], []
    for x in batch:
        p = n - len(x["input_ids"])
        xs.append(x["input_ids"] + [pad_id] * p)
        ys.append(x["labels"] + [-100] * p)
    return {"input_ids": torch.tensor(xs), "labels": torch.tensor(ys)}
