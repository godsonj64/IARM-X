from functools import partial


def build_pretrain_dataset(
    tokenizer,
    dataset_name="HuggingFaceFW/fineweb-edu",
    dataset_config="sample-10BT",
    split="train",
    seq_len=512,
    streaming=True,
    data_files=None,
    pack=True,
):
    """Tokenize text into fixed next-token rows of ``seq_len`` targets.

    With ``pack=True`` (the default) each map batch of documents is joined into
    one stream, one EOS after each document, and cut into windows with
    ``pack_windows``: nothing is truncated, and every row holds ``seq_len``
    targets except the last row of each map batch, whose padding is masked.
    Because rows are fixed-size, ``micro_batch_size`` counts rows and every
    micro-batch has the same shape. With ``pack=False`` each document becomes
    one row truncated to ``seq_len + 1`` tokens.
    """
    from datasets import load_dataset

    if data_files:
        ds = load_dataset(
            "parquet", data_files=data_files, split=split, streaming=streaming
        )
    else:
        ds = load_dataset(
            dataset_name, dataset_config, split=split, streaming=streaming
        )

    ds = ds.select_columns(["text"])  # decode only what is tokenized
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    # A module-level function (not a closure) so DataLoader workers can pickle
    # the dataset under the spawn/forkserver start methods.
    fn = partial(_tokenize_rows, tokenizer=tokenizer, seq_len=seq_len, eos_id=eos_id, pack=pack)
    # The row count changes, so every source column must be dropped.
    columns = ds.column_names or list(next(iter(ds)).keys())
    return ds.map(fn, batched=True, remove_columns=columns)


def encode_texts(tokenizer, texts):
    # split_special_tokens: a literal "<|endoftext|>" or "<|user|>" in web text is
    # ordinary text, not an EOS or a chat-control token.
    return tokenizer(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]


def _tokenize_rows(batch, tokenizer, seq_len, eos_id, pack):
    docs = [ids for ids in encode_texts(tokenizer, batch["text"]) if ids]
    if pack:
        stream = []
        for ids in docs:
            stream.extend(ids)
            stream.append(eos_id)
        return pack_windows(stream, seq_len, eos_id)
    inputs, labels = [], []
    for ids in docs:
        row = pack_windows(ids[: seq_len + 1], seq_len, eos_id)
        inputs.extend(row["input_ids"])
        labels.extend(row["labels"])
    return {"input_ids": inputs, "labels": labels}


def pack_windows(stream, seq_len: int, pad_id: int):
    """Cut a token stream into next-token rows with stride ``seq_len``.

    Row ``i`` predicts ``stream[i*seq_len + 1 : (i+1)*seq_len + 1]`` from the
    tokens before each target, so every token after the first is a target
    exactly once. The last row is padded with ``pad_id`` and its padding labels
    are -100, so no token is dropped and no padding is trained on.
    """
    inputs, labels = [], []
    for start in range(0, max(len(stream) - 1, 0), seq_len):
        chunk = stream[start : start + seq_len + 1]
        x, y = list(chunk[:-1]), list(chunk[1:])
        pad = seq_len - len(x)
        inputs.append(x + [pad_id] * pad)
        labels.append(y + [-100] * pad)
    return {"input_ids": inputs, "labels": labels}


def collate_rows(batch):
    """Stack fixed-length rows from ``build_pretrain_dataset`` into tensors."""
    import torch

    return {
        "input_ids": torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long),
        "labels": torch.tensor([ex["labels"] for ex in batch], dtype=torch.long),
    }
