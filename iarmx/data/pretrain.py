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

    def tokenize(batch):
        ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        return split_documents(ids, seq_len * 4)

    # Splitting changes the row count, so every source column must be dropped.
    columns = ds.column_names or list(next(iter(ds)).keys())
    return ds.map(tokenize, batched=True, remove_columns=columns)


def split_documents(batch_ids, max_len: int):
    """Split tokenized documents into pieces of at most ``max_len`` tokens.

    This bounds what one example presents to the collator without discarding
    anything: ``doc_end`` marks the last piece of each document, so the collator
    emits exactly one boundary token per document.
    """
    pieces, ends = [], []
    for ids in batch_ids:
        for start in range(0, len(ids), max_len):
            pieces.append(ids[start : start + max_len])
            ends.append(start + max_len >= len(ids))
    return {"input_ids": pieces, "doc_end": ends}


def collate_pretrain(batch, pad_id: int, seq_len: int, pack: bool = True):
    """Create next-token examples and mask synthetic trailing padding.

    With ``pack=True`` tokenized documents are concatenated with one EOS/pad
    boundary token after each document, then split into fixed-length blocks.
    Boundary EOS tokens are genuine targets; only synthetic tail padding is
    ignored. Pieces from ``split_documents`` with ``doc_end=False`` continue their
    document, so no boundary token follows them.
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
            if ex.get("doc_end", True):
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
