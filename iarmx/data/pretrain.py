def build_pretrain_dataset(
    tokenizer,
    dataset_name="HuggingFaceFW/fineweb-edu",
    dataset_config="sample-10BT",
    split="train",
    seq_len=512,
    streaming=True,
    data_files=None,
):
    from datasets import load_dataset

    if data_files:
        ds = load_dataset(
            "parquet", data_files=data_files, split=split, streaming=streaming
        )
    else:
        ds = load_dataset(
            dataset_name, dataset_config, split=split, streaming=streaming
        )

    def tokenize(example):
        # Bound the size of any one streamed document presented to the collator.
        # The collator packs across documents and masks only synthetic tail padding.
        return tokenizer(
            example["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len * 4,
        )

    return ds.map(tokenize, remove_columns=None)


def collate_pretrain(batch, pad_id: int, seq_len: int, pack: bool = True):
    """Create next-token examples and mask synthetic trailing padding.

    With ``pack=True`` tokenized documents are concatenated with one EOS/pad
    boundary token between them, then split into fixed-length blocks. Boundary EOS
    tokens are genuine targets; only synthetic tail padding is ignored.
    """
    import torch

    rows, valid_lengths = [], []
    target_len = seq_len + 1

    if pack:
        stream = []
        for ex in batch:
            ids = list(ex.get("input_ids", []))
            if not ids:
                continue
            stream.extend(ids)
            stream.append(pad_id)
        for start in range(0, len(stream), target_len):
            chunk = stream[start : start + target_len]
            if len(chunk) < 2:
                continue
            valid = len(chunk)
            chunk = chunk + [pad_id] * (target_len - valid)
            rows.append(chunk)
            valid_lengths.append(valid)
    else:
        for ex in batch:
            ids = list(ex.get("input_ids", []))[:target_len]
            if len(ids) < 2:
                continue
            valid = len(ids)
            ids = ids + [pad_id] * (target_len - valid)
            rows.append(ids)
            valid_lengths.append(valid)

    if not rows:
        rows = [[pad_id] * target_len]
        valid_lengths = [1]

    full = torch.tensor(rows, dtype=torch.long)
    x = full[:, :-1]
    labels = full[:, 1:].clone()
    for i, valid in enumerate(valid_lengths):
        first_invalid_target = max(valid - 1, 0)
        labels[i, first_invalid_target:] = -100
    return {"input_ids": x, "labels": labels}
