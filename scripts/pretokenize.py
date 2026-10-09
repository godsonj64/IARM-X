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
import threading
import time
from multiprocessing import Pool

import numpy as np

from iarmx.data.memmap import TokenShardWriter, token_dtype
from iarmx.data.pretrain import encode_texts
from iarmx.data.tokenizer import load_tokenizer

_TOKENIZER = None
_EOS = None
_DTYPE = None


def _eos_id(tok) -> int:
    return tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id


def _init_worker(name: str):
    global _TOKENIZER, _EOS, _DTYPE
    # Each worker is single-threaded; parallelism comes from the process pool.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = load_tokenizer(name)
    _EOS, _DTYPE = _eos_id(_TOKENIZER), token_dtype(len(_TOKENIZER))


def _encode(texts: list[str]) -> tuple[np.ndarray, int]:
    """Tokenize a batch into one flat array (EOS after each non-empty document).

    A shard-dtype array pickles at 2 bytes per GPT-2 token instead of ~30 for a
    list of ints, which keeps the single writer process ahead of the workers.
    The writer still range-checks every batch against the vocabulary.
    """
    docs = [ids for ids in encode_texts(_TOKENIZER, texts) if ids]
    total = sum(len(ids) + 1 for ids in docs)
    flat = itertools.chain.from_iterable(itertools.chain(ids, (_EOS,)) for ids in docs)
    return np.fromiter(flat, dtype=_DTYPE, count=total), len(docs)


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
    # Decoding only the text column makes the parquet reader ~3x faster.
    for example in ds.select_columns([args.text_column]):
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
    eos = _eos_id(tok)
    source = {"data_files": args.data_files} if args.data_files else {
        "dataset": args.dataset, "dataset_config": args.dataset_config, "split": args.split,
    }
    writer = TokenShardWriter(
        args.out, vocab_size=len(tok), eos_id=eos, shard_tokens=args.shard_tokens,
        metadata={"tokenizer": args.tokenizer, "source": source, "text_column": args.text_column},
    )

    batches = _batched(_texts(args), args.batch_size)
    pool = None
    # Pool.imap queues finished batches without limit; cap the batches in flight
    # so a slow writer cannot make the main process hold the whole dataset.
    inflight, stop = threading.Semaphore(2 * args.num_proc), threading.Event()

    def throttled():
        for batch in batches:
            inflight.acquire()
            if stop.is_set():
                return
            yield batch

    if args.num_proc > 1:
        pool = Pool(args.num_proc, initializer=_init_worker, initargs=(args.tokenizer,))
        encoded = pool.imap(_encode, throttled())  # imap preserves input order
    else:
        global _TOKENIZER, _EOS, _DTYPE
        # Keep the tokenizer's own multithreaded batch encoding.
        _TOKENIZER, _EOS, _DTYPE = tok, eos, token_dtype(len(tok))
        encoded = map(_encode, batches)

    t0 = last = time.perf_counter()
    try:
        for tokens, n_docs in encoded:
            writer.add_tokens(tokens, n_docs)
            inflight.release()
            now = time.perf_counter()
            if now - last > 30:
                rate = writer.tokens / (now - t0)
                print(f"docs={writer.documents:,} tokens={writer.tokens:,} ({rate:,.0f} tok/s)", flush=True)
                last = now
            if args.max_tokens and writer.tokens >= args.max_tokens:
                break
    finally:
        if pool is not None:
            stop.set()
            inflight.release()  # unblock the pool's feeder thread so terminate() can join it
            pool.terminate()
    index = writer.close()
    print(
        f"wrote {index['tokens']:,} tokens from {index['documents']:,} documents "
        f"in {len(index['shards'])} {index['dtype']} shards to {args.out}"
    )


if __name__ == "__main__":
    main()
