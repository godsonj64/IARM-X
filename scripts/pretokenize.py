"""Tokenize a text dataset once into flat token shards for ``dataset: memmap``.

Every document is kept whole (no truncation) and followed by one EOS token.
Output order equals input order for any ``--num-proc``, so shards are
deterministic. See ``iarmx/data/memmap.py`` for the format.

    # stream FineWeb-Edu sample-10BT from the Hub
    python scripts/pretokenize.py --out data/tokens/fineweb-edu-10bt-gpt2 --num-proc 8

    # or read the parquet files fetched by scripts/download_datasets.py
    python scripts/pretokenize.py --out data/tokens/fineweb-edu-10bt-gpt2 --num-proc 8 \\
        --data-files "data/raw/fineweb-edu-10bt/sample/10BT/*.parquet"
"""

import argparse
import itertools
import os
import time
from multiprocessing import Pool

from iarmx.data.memmap import TokenShardWriter
from iarmx.data.tokenizer import load_tokenizer

_TOKENIZER = None


def _init_worker(name: str):
    global _TOKENIZER
    # Each worker is single-threaded; parallelism comes from the process pool.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = load_tokenizer(name)


def _encode(texts: list[str]) -> list[list[int]]:
    return _TOKENIZER(texts, add_special_tokens=False)["input_ids"]


def _batched(iterable, size: int):
    it = iter(iterable)
    while batch := list(itertools.islice(it, size)):
        yield batch


def _texts(args):
    from datasets import load_dataset

    if args.data_files:
        ds = load_dataset("parquet", data_files=args.data_files, split="train", streaming=True)
    else:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    for example in ds:
        yield example[args.text_column]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory for shards and index.json")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--dataset-config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-files", help="local parquet glob; overrides --dataset")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--batch-size", type=int, default=1000, help="documents per tokenizer call")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--max-tokens", type=int, help="stop after about this many tokens (pilots)")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    source = {"data_files": args.data_files} if args.data_files else {
        "dataset": args.dataset, "dataset_config": args.dataset_config, "split": args.split,
    }
    writer = TokenShardWriter(
        args.out, vocab_size=len(tok), eos_id=eos, shard_tokens=args.shard_tokens,
        metadata={"tokenizer": args.tokenizer, "source": source, "text_column": args.text_column},
    )

    batches = _batched(_texts(args), args.batch_size)
    pool = None
    if args.num_proc > 1:
        pool = Pool(args.num_proc, initializer=_init_worker, initargs=(args.tokenizer,))
        encoded = pool.imap(_encode, batches)  # imap preserves input order
    else:
        global _TOKENIZER
        _TOKENIZER = tok  # keep the tokenizer's own multithreaded batch encoding
        encoded = map(_encode, batches)

    t0 = last = time.perf_counter()
    try:
        for docs in encoded:
            for ids in docs:
                writer.add_document(ids)
            now = time.perf_counter()
            if now - last > 30:
                rate = writer.tokens / (now - t0)
                print(f"docs={writer.documents:,} tokens={writer.tokens:,} ({rate:,.0f} tok/s)", flush=True)
                last = now
            if args.max_tokens and writer.tokens >= args.max_tokens:
                break
    finally:
        if pool is not None:
            pool.terminate()
    index = writer.close()
    print(
        f"wrote {index['tokens']:,} tokens from {index['documents']:,} documents "
        f"in {len(index['shards'])} {index['dtype']} shards to {args.out}"
    )


if __name__ == "__main__":
    main()
