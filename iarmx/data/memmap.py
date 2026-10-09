"""Pre-tokenized token shards for exact, resumable pretraining.

Documents are tokenized once (``scripts/pretokenize.py``), each followed by one
EOS token, and concatenated into flat ``uint16``/``uint32`` shards with an
``index.json``. Nothing is truncated. Training reads fixed windows of
``seq_len + 1`` tokens with stride ``seq_len``, so every token after the first
is a target exactly once per epoch and no target is padding.

The data position is a single integer: the number of windows consumed
globally. ``WindowSampler`` maps global position ``p`` to a window and gives
rank ``r`` of ``w`` the positions ``start + r, start + r + w, ...``. After every
rank has drawn ``k`` windows the union is exactly ``[start, start + w * k)``, so
a run resumes exactly from ``examples_seen`` even with a different GPU count.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

INDEX_FILE = "index.json"
FORMAT_VERSION = 1


def token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size <= 2**16 else np.dtype(np.uint32)


def read_index(path) -> dict:
    index = json.loads((Path(path) / INDEX_FILE).read_text())
    if index.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported token-shard format {index.get('format')!r} in {path}")
    return index


class TokenShardWriter:
    """Append documents to fixed-size token shards; ``close()`` writes the index.

    Shard files and the index are written to temporary names and renamed, so an
    interrupted run never leaves a truncated shard behind a valid index.
    """

    def __init__(self, out_dir, vocab_size: int, eos_id: int, shard_tokens: int = 100_000_000,
                 metadata: dict | None = None):
        if shard_tokens <= 0:
            raise ValueError("shard_tokens must be positive")
        if not 0 <= eos_id < vocab_size:
            raise ValueError(f"eos_id {eos_id} outside vocabulary of size {vocab_size}")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if (self.out_dir / INDEX_FILE).exists():
            raise FileExistsError(f"{self.out_dir / INDEX_FILE} exists; refusing to overwrite")
        self.vocab_size = int(vocab_size)
        self.eos_id = int(eos_id)
        self.dtype = token_dtype(vocab_size)
        self.shard_tokens = int(shard_tokens)
        self.metadata = dict(metadata or {})
        self.shards: list[dict] = []
        self.documents = 0
        self.tokens = 0
        self._buffer: list[np.ndarray] = []
        self._buffered = 0

    def add_document(self, ids) -> int:
        """Append one document plus its EOS separator. Empty documents are skipped."""
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size == 0:
            return 0
        return self.add_tokens(np.append(arr, self.eos_id), documents=1)

    def add_tokens(self, tokens, documents: int) -> int:
        """Append pre-joined documents (each already followed by its EOS)."""
        arr = np.asarray(tokens)
        if arr.size == 0:
            return 0
        if arr.min() < 0 or arr.max() >= self.vocab_size:
            raise ValueError(f"token id outside [0, {self.vocab_size})")
        arr = arr.astype(self.dtype, copy=False)
        self._buffer.append(arr)
        self._buffered += arr.size
        self.documents += int(documents)
        self.tokens += arr.size
        while self._buffered >= self.shard_tokens:
            self._flush(self.shard_tokens)
        return arr.size

    def _flush(self, n: int):
        data = np.concatenate(self._buffer) if len(self._buffer) > 1 else self._buffer[0]
        head, tail = data[:n], data[n:]
        name = f"shard_{len(self.shards):05d}.bin"
        tmp = self.out_dir / (name + ".tmp")
        head.tofile(tmp)
        os.replace(tmp, self.out_dir / name)
        self.shards.append({"file": name, "tokens": int(head.size)})
        self._buffer = [tail] if tail.size else []
        self._buffered = int(tail.size)

    def close(self) -> dict:
        if self._buffered:
            self._flush(self._buffered)
        index = {
            "format": FORMAT_VERSION,
            "dtype": self.dtype.name,
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
            "tokens": self.tokens,
            "documents": self.documents,
            "shards": self.shards,
            **self.metadata,
        }
        tmp = self.out_dir / (INDEX_FILE + ".tmp")
        tmp.write_text(json.dumps(index, indent=2))
        os.replace(tmp, self.out_dir / INDEX_FILE)
        return index


class TokenWindows(Dataset):
    """Next-token windows over pre-tokenized shards: window ``i`` is tokens
    ``[i * seq_len, i * seq_len + seq_len]`` split into inputs and shifted labels."""

    def __init__(self, path, seq_len: int):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.path = Path(path)
        self.seq_len = int(seq_len)
        index = read_index(self.path)
        self.vocab_size = int(index["vocab_size"])
        self.eos_id = int(index["eos_id"])
        self.dtype = np.dtype(index["dtype"])
        shards = [s for s in index["shards"] if s["tokens"] > 0]
        self.files = [self.path / s["file"] for s in shards]
        self.offsets = np.cumsum([0] + [int(s["tokens"]) for s in shards])
        self.total_tokens = int(self.offsets[-1])
        if self.total_tokens != int(index["tokens"]):
            raise ValueError(f"shard sizes in {self.path} do not add up to the indexed token count")
        self.n_windows = (self.total_tokens - 1) // self.seq_len
        if self.n_windows < 1:
            raise ValueError(f"{self.path} holds {self.total_tokens} tokens; need > seq_len={seq_len}")
        self._maps = None

    def __getstate__(self):
        # Memory maps would be pickled as full arrays; reopen lazily in each worker.
        state = self.__dict__.copy()
        state["_maps"] = None
        return state

    def _shard_maps(self):
        if self._maps is None:
            self._maps = [
                np.memmap(f, dtype=self.dtype, mode="r", shape=(int(n),))
                for f, n in zip(self.files, np.diff(self.offsets))
            ]
        return self._maps

    def read(self, start: int, length: int) -> np.ndarray:
        if start < 0 or start + length > self.total_tokens:
            raise IndexError(f"token range [{start}, {start + length}) out of bounds")
        maps, parts = self._shard_maps(), []
        shard = int(np.searchsorted(self.offsets, start, side="right")) - 1
        while length > 0:
            local = start - int(self.offsets[shard])
            take = min(length, int(self.offsets[shard + 1]) - start)
            parts.append(np.asarray(maps[shard][local : local + take], dtype=np.int64))
            start, length, shard = start + take, length - take, shard + 1
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, i):
        if not 0 <= i < self.n_windows:
            raise IndexError(i)
        buf = torch.from_numpy(self.read(int(i) * self.seq_len, self.seq_len + 1))
        return {"input_ids": buf[:-1], "labels": buf[1:]}


class WindowSampler(Sampler):
    """Infinite, rank-partitioned, resumable order over ``n`` windows.

    Global position ``p`` is window ``perm_e[p mod n]`` in epoch ``e = p // n``,
    where ``perm_e`` is a permutation seeded by ``(seed, e)`` (identity when
    ``shuffle=False``). Rank ``r`` of ``world`` yields positions
    ``start + r, start + r + world, ...``.
    """

    def __init__(self, n: int, start: int = 0, world: int = 1, rank: int = 0,
                 shuffle: bool = True, seed: int = 0):
        if n <= 0:
            raise ValueError("n must be positive")
        if not 0 <= rank < world:
            raise ValueError("rank must be in [0, world)")
        if start < 0:
            raise ValueError("start must be non-negative")
        self.n, self.start, self.world, self.rank = int(n), int(start), int(world), int(rank)
        self.shuffle, self.seed = bool(shuffle), int(seed)

    def permutation(self, epoch: int) -> np.ndarray | None:
        if not self.shuffle:
            return None
        return np.random.default_rng([self.seed, epoch]).permutation(self.n)

    def __iter__(self):
        position = self.start + self.rank
        epoch, perm = -1, None
        while True:
            e, i = divmod(position, self.n)
            if e != epoch:
                epoch, perm = e, self.permutation(e)
            yield int(perm[i]) if perm is not None else i
            position += self.world
